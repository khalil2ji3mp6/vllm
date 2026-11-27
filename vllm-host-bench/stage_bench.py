"""Isolated steady-state CPU throughput on Linux, one independent lane per physical core."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
import signal
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from threading import BrokenBarrierError
from time import perf_counter_ns

from cpu_topology import benchmark_topology
from flow_groups import MICRO, MICRO_LAYOUTS
from host_capacity import distribute_all_cores, reserve_output_dir
from runner_report import MEASURED, PHASES, SPEC_ROWS, write_tables
from stage_counters import Counters
from stage_transport import BOUNDARIES, measure_pairs

CASES = {
    "schema": ("Schema 与采样参数", "请求"),
    "template": ("Chat Template", "请求"),
    "tokenize": ("RC5 分词", "请求"),
    "tokenize_native": ("RC5 原生 encode 对照", "请求"),
    "ids_export": ("已有 Encoding 导出 32K Python IDs 对照", "请求"),
    "base64": ("图片 Base64 解码", "图"),
    "jpeg": ("JPEG 解压、方向处理与 RGB", "图"),
    "jpeg_pixels": ("JPEG 仅像素解压对照", "图"),
    "rgb_copy": ("解压后像素复制对照", "图"),
    "encode_image": ("独立图片编码与 Base64", "图"),
    "admit": ("输入校验与前端建状态", "请求"),
    "msgpack": ("前端到引擎的消息编解码", "请求"),
    "drive_scheduler": ("调度、KV、hash 与状态更新", "完整请求"),
    "worker_prefill": ("worker 首段输入准备", "16384-token 步"),
    "worker_decode": ("worker Decode 输入准备", "单序列步"),
    "stream": ("流式反分词、停止检查与序列化", "1024-token 请求"),
    "copy64": ("内存复制对照", "64 MiB 读写"),
}


CASES.update({name: (title, "往返") for name, title in BOUNDARIES.items()})
CASES.update(MICRO)
THREAD_CASES = {
    "schema",
    "template",
    "tokenize",
    "tokenize_native",
    "ids_export",
    "base64",
    "jpeg",
    "jpeg_pixels",
}


def micro_parallelism(stage, layout):
    if layout == "thread_plans":
        if stage == "tokenize_native":
            return "跨请求多核，RC5 原生 encode"
        if stage == "ids_export":
            return "跨进程多核，每进程内导出 Python IDs 受 GIL 约束"
        if stage == "tokenize":
            return "原生 encode 跨请求多核，Python IDs 导出共享进程内 GIL"
    return MICRO_LAYOUTS[layout]


def current_mhz(cpu):
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text()) / 1000
    except (OSError, ValueError):
        return None


def lane(cpu, cases, args, gate, deadline, directory, mask=None):
    # A saturated microbenchmark lane differs from a service role sharing the whole DP mask.
    os.sched_setaffinity(0, mask if mask is not None else {cpu})
    counters = Counters()
    try:
        cases.initialize_lane(args.stages)
        probe_pages = (
            args.stages == ["copy64"] and os.getenv("VLLM_HOST_COPY_PLACEMENT") == "1"
        )
        if probe_pages:
            from memory_placement import placement

            before_copy = {"src": placement(cases.src), "dst": placement(cases.dst)}
        rows = {}
        for stage in args.stages:
            reset = None
            if args.runner_profile:
                _, phase, key = stage.split("/")
                operation = cases.runner_ops[phase][key]
                if phase == "prefill" and key == "worker_batch_update":
                    reset = cases.reset_new_request
                operation()
            else:
                operation = getattr(cases, stage)
                operation(cpu % 32)
            rows[stage] = []
            for _ in range(args.rounds):
                # Wait for the last warmup, then publish one deadline for every NUMA domain.
                if gate.wait() == 0:
                    deadline.value = perf_counter_ns() + int(args.seconds * 1e9)
                gate.wait()
                stop = deadline.value
                mhz = current_mhz(cpu) if mask is None or len(mask) == 1 else None
                pmu_start = counters.read()
                before = resource.getrusage(resource.RUSAGE_THREAD)
                start = perf_counter_ns()
                n = 0
                measured_ns = 0
                if args.runner_profile:
                    while perf_counter_ns() < stop:
                        if reset:
                            reset()
                        begin = perf_counter_ns()
                        value = operation()
                        measured_ns += perf_counter_ns() - begin
                        del value
                        n += 1
                else:
                    while perf_counter_ns() < stop:
                        operation(cpu + n)
                        n += 1
                end = perf_counter_ns()
                after = resource.getrusage(resource.RUSAGE_THREAD)
                pmu = counters.deltas(pmu_start)
                rows[stage].append(
                    dict(
                        n=n,
                        **({"wall_sum_ns": measured_ns} if args.runner_profile else {}),
                        start=start,
                        end=end,
                        user=after.ru_utime - before.ru_utime,
                        system=after.ru_stime - before.ru_stime,
                        faults=after.ru_minflt - before.ru_minflt,
                        major=after.ru_majflt - before.ru_majflt,
                        involuntary=after.ru_nivcsw - before.ru_nivcsw,
                        sampled_mhz=mhz,
                        pmu=pmu,
                        pmu_errors=counters.errors,
                    )
                )
        if probe_pages:
            gate.wait()
            (directory / f"placement-{cpu}.json").write_text(
                json.dumps(
                    {
                        "cpu": cpu,
                        "before": before_copy,
                        "after": {
                            "src": placement(cases.src),
                            "dst": placement(cases.dst),
                        },
                    }
                )
            )
        (directory / f"{cpu}.json").write_text(json.dumps(rows))
    except BaseException as exc:
        if not isinstance(exc, BrokenBarrierError):
            traceback.print_exc()
        gate.abort()
        raise
    finally:
        counters.close()


def node(cpus, args, gate, deadline, directory):
    # Each NUMA group first-touches its own model metadata, corpus and image.
    os.setsid()
    os.sched_setaffinity(0, {cpus[0]})
    processes = []
    try:
        from stage_cases import Cases
        from vllm.utils.gc_utils import freeze_gc_heap

        cases = Cases(args.stages, len(cpus) if args.threaded else 1)
        fixture = {"token_ids_sha256": cases.ids_hash.hexdigest()}
        if not args.runner_profile:
            fixture["image_sha256"] = hashlib.sha256(cases.raw).hexdigest()
        (directory / f"fixture-{cpus[0]}.json").write_text(json.dumps(fixture))
        freeze_gc_heap()
        if args.runner_profile:
            import torch
            torch.set_num_threads(1)
            # A TP1 worker processes one scheduled batch on its DP mask.
            lane(cpus[0], cases, args, gate, deadline, directory, mask=cpus)
            return
        if not args.threaded and len(list(Path("/proc/self/task").iterdir())) != 1:
            raise RuntimeError(
                "CPU stage fixture must be single-threaded before forking"
            )
        if args.threaded:
            with ThreadPoolExecutor(max_workers=len(cpus)) as threads:
                jobs = [
                    threads.submit(lane, cpu, cases, args, gate, deadline, directory)
                    for cpu in cpus
                ]
                for job in jobs:
                    job.result()
            return
        for cpu in cpus:
            p = mp.get_context("fork").Process(
                target=lane, args=(cpu, cases, args, gate, deadline, directory)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            if p.exitcode:
                raise RuntimeError(f"Stage worker {p.pid} exited {p.exitcode}")
    except BaseException:
        gate.abort()
        raise
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()


def sample(layout, args, out):
    ctx = mp.get_context("spawn")
    lane_ids = [group[0] for group in layout] if args.runner_profile else [c for group in layout for c in group]
    gate = ctx.Barrier(len(lane_ids), timeout=300)
    deadline = ctx.RawValue("q", 0)
    with TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        processes = []
        try:
            for cpus in layout:
                p = ctx.Process(
                    target=node, args=(cpus, args, gate, deadline, directory)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                if p.exitcode:
                    raise RuntimeError(f"NUMA worker {p.pid} exited {p.exitcode}")
        finally:
            for p in processes:
                if p.is_alive():
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        p.terminate()
            for p in processes:
                p.join()
        fixtures = [json.loads(p.read_text()) for p in directory.glob("fixture-*.json")]
        assert len(fixtures) == len(layout) and all(f == fixtures[0] for f in fixtures)
        placements = [
            json.loads(p.read_text()) for p in directory.glob("placement-*.json")
        ]
        lanes = [
            json.loads((directory / f"{cpu}.json").read_text())
            for cpu in lane_ids
        ]
    results = {}
    for stage in args.stages:
        rounds = []
        for group in zip(*(s[stage] for s in lanes), strict=True):
            n = sum(r["n"] for r in group)
            if not n:
                raise RuntimeError(
                    "All lanes missed the sample window. Increase --seconds."
                )
            seconds = (
                max(r["end"] for r in group) - min(r["start"] for r in group)
            ) / 1e9
            cpu_s = sum(r["user"] + r["system"] for r in group)
            mhz = [r["sampled_mhz"] for r in group if r["sampled_mhz"] is not None]
            pmu = {
                key: sum(r["pmu"][key] for r in group) / n
                for key in ("cycles", "instructions", "cache_misses")
                if all(key in r["pmu"] for r in group)
            }
            rounds.append(
                dict(
                    pmu_per_op=pmu,
                    pmu_errors={
                        k: v for r in group for k, v in r["pmu_errors"].items()
                    },
                    active_lanes=sum(r["n"] > 0 for r in group),
                    operations=n,
                    **({"wall_sum_ns": sum(r["wall_sum_ns"] for r in group)} if args.runner_profile else {}),
                    wall_s=seconds,
                    ops_s=n / seconds,
                    effective_cores=cpu_s / seconds,
                    cpu_us_per_op=cpu_s * 1e6 / n,
                    system_us_per_op=sum(r["system"] for r in group) * 1e6 / n,
                    minor_faults_per_op=sum(r["faults"] for r in group) / n,
                    major_faults=sum(r["major"] for r in group),
                    involuntary_switches=sum(r["involuntary"] for r in group),
                    sampled_mhz_median=median(mhz) if mhz else None,
                )
            )
        results[stage] = {
            "rounds": rounds,
            "ops_s": median(r["ops_s"] for r in rounds),
            "effective_cores": median(r["effective_cores"] for r in rounds),
        }
    plan = dict(cpu_masks=layout, fixture=fixtures[0], stages=results)
    if placements:
        plan["copy_placement"] = placements
    return plan


def compare(paths):
    first, second = [json.loads(path.read_text()) for path in paths]
    for field in ("versions", "vllm_source", "seconds", "rounds", "protocol"):
        if first.get(field) != second.get(field):
            raise ValueError(f"Two runs differ in {field}")
    tables = []
    for key, title in (
        ("plans", "逐核进程"),
        ("thread_plans", "共享解释器"),
        ("ipc_plans", "进程间传输"),
    ):
        left, right = first.get(key, {}), second.get(key, {})
        if not left and not right:
            continue
        if "1" not in left or "1" not in right:
            raise ValueError("Both runs must include a single-core case")
        a, b = max(map(int, left)), max(map(int, right))
        plans = (left["1"], right["1"], left[str(a)], right[str(b)])
        if any(
            p.get("fixture") != plans[0].get("fixture")
            or p["stages"].keys() != plans[0]["stages"].keys()
            for p in plans[1:]
        ):
            raise ValueError("The stage suites or input fingerprints differ")
        lines = [
            f"### {title}",
            "",
            f"| 流程 | 多核配置的并行位置 | 单核吞吐 B/A | 最大实测吞吐 B({b}核)/A({a}核) |",
            "|---|---|---:|---:|",
        ]
        for name in plans[0]["stages"]:
            lo_a, lo_b, hi_a, hi_b = (p["stages"][name]["ops_s"] for p in plans)
            lines.append(
                f"| {CASES[name][0]} | {micro_parallelism(name, key)} | {lo_b / lo_a:.2f}× | {hi_b / hi_a:.2f}× |"
            )
        tables.append("\n".join(lines))
    print("\n\n".join(tables))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-profile", action="store_true", help="Isolate host runner operations on fixed scheduler steps")
    parser.add_argument("--core-sweep", nargs="+", default=None)
    parser.add_argument("--stages", nargs="+", choices=CASES)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("process", "threads", "ipc"),
        default=("process", "threads", "ipc"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("stages"))
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A_JSON", "B_JSON"))
    parser.add_argument("--keep-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.runner_profile and (args.stages or args.compare):
        parser.error("--runner-profile selects its own operations and report")
    explicit = args.stages is not None
    args.stages = args.stages or ["text_request", "image_request"]
    if args.compare:
        try:
            compare(args.compare)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        return
    if args.rounds < 1 or not 0 < args.seconds < 240:
        parser.error("Use positive rounds and a sample window below 240 seconds")
    if args.core_sweep is None:
        args.core_sweep = ["1"] if args.runner_profile else ["1", "8", "16", "32"]
    if args.runner_profile and args.core_sweep != ["1"]:
        parser.error("The isolated worker profile measures 1 physical core; use --core-sweep 1")
    topo = benchmark_topology()
    physical = set(topo["physical_core_cpus"])
    # Spread each core count across available NUMA domains, one SMT sibling per core.
    pools = [[c for c in node if c in physical] for node in topo["numa_cpus"]]
    ordered = [
        pool[i] for i in range(max(map(len, pools))) for pool in pools if i < len(pool)
    ]
    try:
        counts = list(
            dict.fromkeys(
                len(physical) if n == "all" else int(n) for n in args.core_sweep
            )
        )
    except ValueError:
        parser.error("Core counts must be integers or all")
    if any(n < 1 or n > len(physical) for n in counts):
        parser.error(f"Use between 1 and {len(physical)} physical cores")
    # Apply the host benchmark's CPU-only settings before heavyweight libraries are imported.
    from importlib.metadata import version

    import pybase64
    from PIL import features

    os.environ["RAYON_NUM_THREADS"] = "1"
    out = reserve_output_dir(args.out)
    data = dict(
        protocol=1,
        available_physical_cores=len(physical),
        machine=platform.machine(),
        kernel=platform.release(),
        python=platform.python_version(),
        libc=os.confstr("CS_GNU_LIBC_VERSION"),
        page_bytes=os.sysconf("SC_PAGE_SIZE"),
        jpeg_turbo=features.version_feature("libjpeg_turbo"),
        base64_runtime=pybase64.get_version(),
        decoder_environment={
            k: v for k, v in os.environ.items() if k.startswith(("JSIMD_", "PILLOW_"))
        },
        vllm_source=os.environ.get("VLLM_HOST_SOURCE_COMMIT"),
        versions={
            p: version(p)
            for p in ("tokenizers", "pillow", "pybase64", "transformers", "torch")
        },
        seconds=args.seconds,
        rounds=args.rounds,
        plans={},
        thread_plans={},
        ipc_plans={},
    )
    if args.runner_profile:
        args.stages = [f"runner/{phase}/{key}" for phase in PHASES for key in MEASURED]
        args.stages.extend(f"runner/speculative/{key}" for _, _, key in SPEC_ROWS)
        if args.keep_diagnostics:
            args.stages.append("runner/bookkeeping/parse_verified_ids")
        args.threaded = False
        profiles = {}
        layout = [[ordered[0]]]
        plan = sample(layout, args, out)
        profiles[1] = {"cpu": ordered[0], "dp_size": 1, "tp_size": 1}
        for stage, result in plan["stages"].items():
            _, phase, key = stage.split("/")
            rounds = result["rounds"]
            if any(r["active_lanes"] != len(layout) for r in rounds):
                raise RuntimeError(f"{stage}: some DP workers missed the window; increase --seconds")
            profiles[1].setdefault(phase, {})[key] = {
                "wall_sum_ns": sum(r["wall_sum_ns"] for r in rounds),
                "calls": sum(r["operations"] for r in rounds),
                "ops_s": result["ops_s"],
            }
        if args.keep_diagnostics:
            data["plans"][1] = plan
            (out / "stages.json").write_text(json.dumps(data, indent=2) + "\n")
        print(write_tables(profiles, out), end="")
        return
    reports = []
    for mode, name, heading in (
        ("process", "plans", "独立进程，每物理核一条流水"),
        ("threads", "thread_plans", "共享解释器，最多八个进程，每核一个线程"),
        ("ipc", "ipc_plans", "进程间传输，每 DP 一对收发进程"),
    ):
        if mode not in args.layouts or (
            mode == "threads" and not (explicit or args.keep_diagnostics)
        ):
            continue
        stages = [
            s
            for s in args.stages
            if (
                (mode == "ipc" and s in BOUNDARIES)
                or (mode == "threads" and s in THREAD_CASES)
                or (mode == "process" and s not in BOUNDARIES)
            )
        ]
        if not stages:
            continue
        trial = argparse.Namespace(
            **(vars(args) | dict(stages=stages, threaded=mode == "threads"))
        )
        for count in counts:
            selected = set(ordered[:count])
            layout = [[c for c in node if c in selected] for node in pools]
            layout = [g for g in layout if g]
            if mode != "process":
                layout = distribute_all_cores(
                    {i: deque(g) for i, g in enumerate(layout)}, min(8, count)
                )
            data[name][count] = (
                measure_pairs(layout, trial, out)
                if mode == "ipc"
                else sample(layout, trial, out)
            )
            (out / "stages.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            )
        table = [
            "### " + heading,
            "",
            "| 单流程 | 单位 | 多核配置的并行位置 | "
            + " | ".join(f"{c} 核 吞吐/s（有效核）" for c in counts)
            + " |",
            "|---|---|---|" + "---:|" * len(counts),
        ]
        for stage in stages:
            values = [data[name][n]["stages"][stage] for n in counts]
            table.append(
                "| "
                + " | ".join(
                    [
                        *CASES[stage], micro_parallelism(stage, name),
                        *(
                            f"{v['ops_s']:,.2f}（{v['effective_cores']:.1f}）"
                            for v in values
                        ),
                    ]
                )
                + " |"
            )
        reports.append("\n".join(table))
    report = "\n\n".join(reports) + "\n"
    if not explicit:
        if args.keep_diagnostics:
            (out / "stages-details.md").write_text(report)
        header = (
            ["流程", "工作单位", "多核配置的并行位置"]
            + [f"{n} 核吞吐/s" for n in counts]
            + ["扩核倍数", f"{max(counts)} 核有效核"]
        )
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for stage in args.stages:
            label, unit = CASES[stage]
            plan = data["plans"]
            if not plan:
                continue
            values = [plan[n]["stages"][stage]["ops_s"] for n in counts]
            peak = plan[max(counts)]["stages"][stage]
            ratio = (
                f"{peak['ops_s'] / plan[1]['stages'][stage]['ops_s']:.2f}×"
                if 1 in counts
                else "—"
            )
            cells = [
                label,
                unit,
                MICRO_LAYOUTS["plans"],
                *(f"{v:,.2f}" for v in values),
                ratio,
                f"{peak['effective_cores']:.1f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        report = "\n".join(lines) + "\n"
    (out / "stages.md").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
