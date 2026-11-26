"""One-shot Linux copy64, CPU affinity and NUMA diagnosis."""

import argparse
import ctypes
import json
import mmap
import multiprocessing as mp
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from itertools import count
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
from cpu_topology import cpu_topology
from memory_placement import bind_memory, placement

MIB = 1 << 20
NS = 10**9
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_files():
    """Include enclosing cgroup limits, even when the leaf has no quota."""
    result = set()
    groups = [
        s.split(":", 2) for s in Path("/proc/self/cgroup").read_text().splitlines()
    ]
    for mount in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = mount.split(" - ")
        fs, _, flags = right.split()[:3]
        if fs not in ("cgroup", "cgroup2"):
            continue
        root, mountpoint = map(Path, left.split()[3:5])
        for _, ctrls, group in groups:
            options = set(ctrls.split(","))
            if (fs == "cgroup2" and ctrls) or (
                fs == "cgroup" and not options.intersection(flags.split(","))
            ):
                continue
            try:
                path = mountpoint / Path(group).relative_to(root)
            except ValueError:
                continue
            while True:
                for name in (
                    "cpu.stat",
                    "cpu.max",
                    "cpu.cfs_quota_us",
                    "cpu.cfs_period_us",
                    "memory.max",
                    "memory.current",
                    "memory.limit_in_bytes",
                    "memory.usage_in_bytes",
                ):
                    if (path / name).is_file():
                        result.add(path / name)
                if path == mountpoint:
                    break
                path = path.parent
    return sorted(result)


def cpu_stats(files):
    return {
        str(p): dict(line.split() for line in content.splitlines())
        for p in files
        if p.name == "cpu.stat" and (content := read(p)) and "nr_throttled" in content
    }


def buffers(size, allocator):
    if allocator == "numpy":
        src = np.full(size, 113, np.uint8)
        dst = np.zeros_like(src)
        dst.fill(1)
    else:

        def mapped():
            region = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            return np.frombuffer(region, dtype=np.uint8)

        src, dst = mapped(), mapped()
        src.fill(113)
        dst.fill(1)
    return src, dst


def worker(pipe, cpu, node, touch_cpu, alloc, size, strict):
    try:
        os.sched_setaffinity(0, {touch_cpu})
        if strict:
            error = bind_memory(node)
            if error:
                pipe.send({"skip": error})
                return
        src, dst = buffers(size, alloc)
        os.sched_setaffinity(0, {cpu})
        assert os.sched_getaffinity(0) == {cpu}

        def locations():
            return {"src": placement(src), "dst": placement(dst)}

        memcpy = ctypes.CDLL(None).memcpy
        memcpy.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
        memcpy.restype = ctypes.c_void_p
        ptrs = dst.ctypes.data, src.ctypes.data, size
        zero = np.uint8(0)
        operations = {
            "numpy": lambda: np.copyto(dst, src),
            "fixed": lambda: np.copyto(dst, src),
            "libc": lambda: memcpy(*ptrs),
            "xor0": lambda: np.bitwise_xor(src, zero, out=dst),
        }
        for name in ("numpy", "libc", "xor0"):
            dst[0] = dst[-1] = 1
            operations[name]()
            if dst[0] != 113 or dst[-1] != 113:
                raise RuntimeError(f"{name}: copied bytes do not match source")
        pipe.send(
            {
                "cpu": cpu,
                "node": node,
                "touch_cpu": touch_cpu,
                "before": locations(),
                "src_page_offset": src.ctypes.data % mmap.PAGESIZE,
                "dst_page_offset": dst.ctypes.data % mmap.PAGESIZE,
            }
        )
        while True:
            command = pipe.recv()
            if command == "finish":
                good = int(dst.min()) == int(dst.max()) == 113
                pipe.send({"after": locations(), "valid": good})
                break
            name, start_at, duration, fixed_count = command
            op = operations[name]
            delay = (start_at - time.monotonic_ns()) / NS
            if delay > 0:
                time.sleep(delay)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            start = time.monotonic_ns()
            if fixed_count:
                for _ in range(fixed_count):
                    op()
                n = fixed_count
            else:
                n = 0
                while time.monotonic_ns() < start_at + duration:
                    op()
                    n += 1
            end = time.monotonic_ns()
            after = resource.getrusage(resource.RUSAGE_SELF)
            pipe.send(
                {
                    "n": n,
                    "start": start,
                    "end": end,
                    "user": after.ru_utime - usage.ru_utime,
                    "sys": after.ru_stime - usage.ru_stime,
                    "faults": after.ru_minflt - usage.ru_minflt,
                    "major": after.ru_majflt - usage.ru_majflt,
                }
            )
    except (OSError, ValueError, RuntimeError, MemoryError, AssertionError):
        pipe.send({"error": traceback.format_exc()})
    finally:
        pipe.close()


def collect(pipes):
    pending = set(pipes)
    results = {}
    deadline = time.monotonic() + 180
    while pending:
        ready = wait(pending, max(0, deadline - time.monotonic()))
        if not ready:
            raise TimeoutError("timed out waiting for copy workers")
        for p in ready:
            try:
                row = p.recv()
            except EOFError as exc:
                raise RuntimeError(
                    "copy worker exited before sending its result"
                ) from exc
            if "error" in row:
                raise RuntimeError(row["error"])
            results[p] = row
            pending.remove(p)
    return [results[p] for p in pipes]


def locality(lanes, phase, key):
    counts = Counter()
    for lane in lanes:
        obs = lane[phase][key]["nodes"]
        counts["local"] += obs.get(str(lane["node"]), 0)
        counts["unknown"] += obs.get("?", 0)
        counts["total"] += sum(obs.values())
    return dict(counts)


def run_case(label, layout, alloc, size, strict, modes, args, files):
    ctx = mp.get_context("fork")
    processes, pipes = [], []
    try:
        for cpu, node, touch in layout:
            conn, child = ctx.Pipe()
            proc = ctx.Process(
                target=worker, args=(child, cpu, node, touch, alloc, size, strict)
            )
            proc.start()
            child.close()
            pipes.append(conn)
            processes.append(proc)
        lanes = collect(pipes)
        errors = sorted({r["skip"] for r in lanes if "skip" in r})
        if errors:
            return {"label": label, "skipped": errors}
        results = {}
        fixed = None
        for name in modes:
            rounds = []
            for _ in range(args.rounds):
                begin_stats = cpu_stats(files)
                start_at = time.monotonic_ns() + 250_000_000
                for i, p in enumerate(pipes):
                    p.send(
                        (
                            name,
                            start_at,
                            int(args.seconds * NS),
                            fixed[i] if name == "fixed" else 0,
                        )
                    )
                rows = collect(pipes)
                wall = (
                    max(r["end"] for r in rows) - min(r["start"] for r in rows)
                ) / NS
                n = sum(r["n"] for r in rows)
                if not n:
                    raise RuntimeError("All lanes missed the timing window")
                if name == "numpy":
                    fixed = [max(1, r["n"]) for r in rows]
                cores = sum(r["user"] + r["sys"] for r in rows) / wall
                stats = cpu_stats(files)
                throttled = {
                    k: int(v.get("nr_throttled", 0))
                    - int(begin_stats[k].get("nr_throttled", 0))
                    for k, v in stats.items()
                    if k in begin_stats
                }
                rounds.append(
                    {
                        "gib_s": n * 2 * size / (1 << 30) / wall,
                        "effective_cores": cores,
                        "start_skew_ms": (
                            max(r["start"] for r in rows)
                            - min(r["start"] for r in rows)
                        )
                        / 1e6,
                        "max_late_ms": (max(r["start"] for r in rows) - start_at) / 1e6,
                        "throttle_events": throttled,
                        "wall_s": wall,
                        "lanes": rows,
                    }
                )
            results[name] = rounds
        for p in pipes:
            p.send("finish")
        for lane, end in zip(lanes, collect(pipes), strict=True):
            if not end["valid"]:
                raise RuntimeError("copy data check failed")
            lane.update(end)
        return {
            "label": label,
            "allocator": alloc,
            "bytes_per_buffer": size,
            "lanes": lanes,
            "results": results,
        }
    finally:
        for p in pipes:
            p.close()
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()


def legacy(out, args):
    log = out / "legacy.log"
    with log.open("w") as stream:
        proc = subprocess.run(
            [
                str(ROOT / "run-stages.sh"),
                "--layouts",
                "process",
                "--stages",
                "copy64",
                "--out",
                str(out / "legacy"),
                "--rounds",
                str(args.rounds),
                "--seconds",
                str(args.seconds),
            ],
            env=os.environ | {"VLLM_HOST_COPY_PLACEMENT": "1"},
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode:
        return {"error": f"exit {proc.returncode}, see {log.name}"}
    data = json.loads((out / "legacy/stages.json").read_text())
    node_for_cpu = {
        c: n for n, cpus in cpu_topology()["numa_nodes"].items() for c in cpus
    }
    result = {}
    for k, v in data["plans"].items():
        lanes = v["copy_placement"]
        if len(lanes) != int(k):
            raise RuntimeError("Missing page observations in original copy64 probe")
        for lane in lanes:
            lane["node"] = node_for_cpu[lane["cpu"]]
        result[k] = v["stages"]["copy64"] | {"lanes": lanes}
    return result


def summarize(data):
    lines = [
        "# copy64 内存诊断",
        "",
        (
            f"{data['arch']}，{data['physical_cores']} 个可用物理核。"
            "每轮按实际完成量和墙钟计算，GiB/s 包含读和写。"
        ),
        "",
        "| 场景 | 核数 | 复制方式 | GiB/s | 有效核 | 源 / 目标本地页（测前 → 测后） |",
        "|---|---:|---|---:|---:|---|",
    ]

    def local(row, phase):
        values = []
        for key in ("src", "dst"):
            c = locality(row["lanes"], phase, key)
            if c["unknown"] == c["total"]:
                values.append("未知")
                continue
            val = f"{100 * c['local'] / c['total']:.1f}%"
            if c["unknown"]:
                val += f"（未知 {100 * c['unknown'] / c['total']:.0f}%）"
            values.append(val)
        return "/".join(values)

    legacy_result = data.get("legacy", {})
    if "error" in legacy_result:
        lines.append(f"| 原程序 | — | {legacy_result['error']} | — | — | — |")
    else:
        for n, v in legacy_result.items():
            lines.append(
                f"| 原程序单独跑 copy64 | {n} | numpy | {v['ops_s'] / 16:.2f} | "
                f"{v['effective_cores']:.1f} | {local(v, 'before')} → {local(v, 'after')} |"
            )

    def med(rounds, key):
        return statistics.median(r[key] for r in rounds)

    index = {}
    for row in data["cases"]:
        if "skipped" in row:
            lines.append(
                f"| {row['label']} | — | 跳过 {' '.join(row['skipped'])} | — | — | — |"
            )
            continue
        location = local(row, "before") + " → " + local(row, "after")
        for name, rounds in row["results"].items():
            speed = med(rounds, "gib_s")
            index[row["label"], name] = speed
            lines.append(
                f"| {row['label']} | {len(row['lanes'])} | {name} | {speed:.2f} | "
                f"{med(rounds, 'effective_cores'):.1f} | {location} |"
            )
    base = index.get(("全机本地", "numpy"))
    if base:
        lines += ["", "全机对照的吞吐倍率。分母为全机本地 NumPy。", ""]
        for label, mode in [
            ("全机本地", "fixed"),
            ("全机本地", "libc"),
            ("全机本地", "xor0"),
            ("全机显式绑内存", "numpy"),
            ("全机匿名 mmap", "numpy"),
            ("全机异节点首次写页", "numpy"),
        ]:
            if (label, mode) in index:
                lines.append(f"- {label} {mode} 为 {index[label, mode] / base:.2f}×")
        separate = sum(
            v
            for (label, mode), v in index.items()
            if label.startswith("单节点 ") and mode == "numpy"
        )
        if separate:
            lines.append(
                f"- 全机同时运行 / 各节点分别运行吞吐之和为 {base / separate:.2f}×"
            )
        v = legacy_result.get(str(data["physical_cores"]))
        if v:
            lines.append(
                f"- 原计时框架单跑 / 独立 NumPy 为 {v['ops_s'] / 16 / base:.2f}×"
            )
        full = next(r for r in data["cases"] if r["label"] == "全机本地")
        rounds = full["results"]["numpy"]
        throttle = max(
            (v for r in rounds for v in r["throttle_events"].values()), default="未知"
        )
        sys_percent = statistics.median(
            100
            * sum(w["sys"] for w in r["lanes"])
            / max(1e-9, sum(w["sys"] + w["user"] for w in r["lanes"]))
            for r in rounds
        )
        faults = statistics.median(sum(w["faults"] for w in r["lanes"]) for r in rounds)
        major = statistics.median(sum(w["major"] for w in r["lanes"]) for r in rounds)
        lines += [
            "",
            (
                f"全机本地 NumPy 的启动偏斜中位数为 {med(rounds, 'start_skew_ms'):.2f} ms。"
                f"单轮最大 cgroup 节流事件增量为 {throttle}。"
                f"祖先计数可包含同组其他进程。复制窗口内进程内核态 CPU 占比 {sys_percent:.1f}%，"
                f"每轮次缺页 / 主缺页中位数 {faults:.0f} / {major:.0f}。"
            ),
        ]
        groups = {}
        for i, lane in enumerate(full["lanes"]):
            groups.setdefault(lane["node"], []).append(i)
        lines += [
            "",
            "| 执行节点 | 物理核 | 节点独跑 GiB/s | 全机同时跑时该节点 GiB/s |",
            "|---:|---:|---:|---:|",
        ]
        for node, indices in groups.items():
            bw = [
                sum(r["lanes"][i]["n"] for i in indices)
                * 2
                * full["bytes_per_buffer"]
                / (1 << 30)
                / r["wall_s"]
                for r in rounds
            ]
            solo = index[f"单节点 {node} 全核", "numpy"]
            lines.append(
                f"| {node} | {len(indices)} | {solo:.2f} | {statistics.median(bw):.2f} |"
            )
        lines += [
            "",
            "fixed 为预先确定次数、循环内不读时钟的 NumPy 复制；xor0 为同字节量的独立 ufunc 读写对照。",
            "本地率按执行 CPU 所在节点计算。未知页单列。不同场景均重新建进程和缓冲区，同一行布局的复制方式复用缓冲区。",
        ]
        lines += ["", "### 本次可核实的信号", ""]
        old = legacy_result.get(str(data["physical_cores"]))
        if old:
            ratio = old["ops_s"] / 16 / base
            if 0.8 <= ratio <= 1.2:
                lines.append("- 原框架和独立计时的全核吞吐相差在 20% 以内。")
            else:
                lines.append(
                    f"- 原框架 / 独立计时为 {ratio:.2f}×，应结合原框架页落点、固定次数对照和启动偏斜判断框架影响。"
                )
            samples = [
                locality(old["lanes"], phase, buf)
                for phase in ("before", "after")
                for buf in ("src", "dst")
            ]
            if any(c["unknown"] > 0 for c in samples):
                lines.append(
                    "- 原框架有页无法确定归属，本机对页放置的结论只覆盖已定位样本。"
                )
            elif all(c["local"] >= 0.95 * c["total"] for c in samples):
                lines.append(
                    "- 原框架的源和目标在前后两次采样中均至少 95% 位于执行节点，大面积远端放置缺少证据。"
                )
            else:
                lines.append(
                    "- 原框架观察到超过 5% 的非本地页，优先比较显式绑内存后的本地率和带宽。"
                )
        for label, mode, desc in [
            ("全机本地", "fixed", "去掉循环内时钟读取"),
            ("全机本地", "libc", "切换 libc memcpy"),
            ("全机本地", "xor0", "切换独立的逐字节 ufunc"),
            ("全机显式绑内存", "numpy", "强制在执行节点分配"),
            ("全机匿名 mmap", "numpy", "切换私有匿名映射"),
        ]:
            ratio = index.get((label, mode), 0) / base
            if ratio >= 1.25:
                lines.append(f"- {desc}后吞吐提高到 {ratio:.2f}×，值得单独归因。")
        if separate and base < 0.75 * separate:
            lines.append(
                "- 多节点同时跑的总量低于各节点分别运行之和的 75%，请看上表哪些节点在并跑时下降。"
            )
        lines.append(
            f"- NumPy {data['numpy']}，{data['libc']}，基础页 {data['page_bytes'] // 1024} KiB。"
            + " ".join(data["allowed"])
        )
    return "\n".join(lines) + "\n"


def memory_available(files):
    limits = [
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    ]
    for p in files:
        current = {
            "memory.max": "memory.current",
            "memory.limit_in_bytes": "memory.usage_in_bytes",
        }.get(p.name)
        limit = read(p)
        usage = read(p.with_name(current)) if current else None
        if current and limit and limit.isdecimal() and usage:
            limits.append(int(limit) - int(usage))
    return min(limits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("memory-diag"))
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    if not (0.1 <= args.seconds <= 60 and 1 <= args.rounds <= 20):
        ap.error("use 0.1–60 seconds and 1–20 rounds")
    topo = cpu_topology()
    physical = set(topo["physical_core_cpus"])
    nodes = {
        n: [c for c in cpus if c in physical] for n, cpus in topo["numa_nodes"].items()
    }
    nodes = {n: cs for n, cs in nodes.items() if cs}
    if not nodes:
        ap.error("NUMA node cpulists unavailable in sysfs")
    files = cgroup_files()
    if memory_available(files) < (len(physical) * 64 + 512) * MIB:
        ap.error("insufficient available RAM for 64 MiB per physical core plus 512 MiB")
    for i in count():
        out = (
            args.out.absolute()
            if not i
            else args.out.absolute().with_name(f"{args.out.name}-{i}")
        )
        try:
            out.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    node_ids = list(nodes)
    local = [(c, n, c) for n, cs in nodes.items() for c in cs]
    spread = [(cs[0], n, cs[0]) for n, cs in nodes.items()]
    cases = [("单核", local[:1]), ("每节点一核", spread)]
    cases.extend(
        (f"单节点 {n} 全核", [(c, n, c) for c in cs]) for n, cs in nodes.items()
    )
    if len(nodes) > 1:
        width = min(len(cs) for cs in nodes.values())
        # Compare the same number of cores concentrated vs spread over nodes.
        balanced = [(cs[i], n, cs[i]) for i in range(width) for n, cs in nodes.items()]
        cases.append(("等核数跨节点", balanced[:width]))
    data = {
        "arch": platform.machine(),
        "kernel": platform.release(),
        "numpy": np.__version__,
        "libc": os.confstr("CS_GNU_LIBC_VERSION"),
        "physical_cores": len(physical),
        "page_bytes": mmap.PAGESIZE,
        "node_memory": {
            n: read(f"/sys/devices/system/node/node{n}/meminfo") for n in nodes
        },
        "node_distances": {
            n: read(f"/sys/devices/system/node/node{n}/distance") for n in nodes
        },
        "nodes": nodes,
        "seconds": args.seconds,
        "rounds": args.rounds,
        "sys_settings": {
            p: read(p)
            for p in (
                "/proc/sys/kernel/numa_balancing",
                "/sys/kernel/mm/transparent_hugepage/enabled",
                "/proc/self/cgroup",
                "/sys/devices/system/clocksource/clocksource0/current_clocksource",
            )
        },
        "allowed": [
            s
            for s in Path("/proc/self/status").read_text().splitlines()
            if s.startswith(("Cpus_allowed_list", "Mems_allowed_list"))
        ],
        "cgroups": {str(p): read(p) for p in files},
        "cases": [],
    }

    def save():
        (out / "report.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )

    def measure(
        label, layout, alloc="numpy", size=32 * MIB, strict=False, modes=("numpy",)
    ):
        print(label, file=sys.stderr, flush=True)
        data["cases"].append(
            run_case(label, layout, alloc, size, strict, modes, args, files)
        )
        save()

    for label, layout in cases:
        measure(label, layout)
    measure("全机本地", local, modes=("numpy", "fixed", "libc", "xor0"))
    measure("全机显式绑内存", local, strict=True)
    measure("全机匿名 mmap", local, alloc="mmap")
    measure("全机小缓冲区 32 KiB", local, size=32 * 1024)
    if len(nodes) > 1:
        remote = []
        for j, n in enumerate(node_ids):
            peers = nodes[node_ids[(j + 1) % len(nodes)]]
            remote.extend((c, n, peers[i % len(peers)]) for i, c in enumerate(nodes[n]))
        measure("全机异节点首次写页", remote)
    print("原框架单独跑 copy64", file=sys.stderr, flush=True)
    data["legacy"] = legacy(out, args)
    save()
    report = summarize(data)
    (out / "report.md").write_text(report)
    print(report, end="")
    print(f"结果保存在 {out}", file=sys.stderr)
    if "error" in data["legacy"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
