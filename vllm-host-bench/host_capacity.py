#!/usr/bin/env python3
"""Run synchronized vLLM Host workloads across disjoint physical CPU sets."""
import argparse
import csv
import json
import math
import os
import signal
import shutil
import subprocess
from collections import deque
from itertools import count
from pathlib import Path
from statistics import median
from time import monotonic, sleep

from bench import frontend_plan
from cpu_topology import cpu_topology
from report import write_stage_tables

STANDARD_CORE_SWEEP = (48, 96, 192)


def distribute_all_cores(pools, count):
    if count < len(pools):
        groups = [[] for _ in range(count)]
        for pool in sorted(pools.values(), key=len, reverse=True):
            min(groups, key=len).extend(pool)
        return groups

    replicas_per_node = dict.fromkeys(pools, 1)
    for _ in range(count - len(pools)):
        eligible = [node for node in pools if replicas_per_node[node] < len(pools[node])]
        node = max(eligible, key=lambda i: len(pools[i]) / (replicas_per_node[i] + 1))
        replicas_per_node[node] += 1
    groups = []
    for node, pool in pools.items():
        width, remainder = divmod(len(pool), replicas_per_node[node])
        for i in range(replicas_per_node[node]):
            groups.append([pool.popleft() for _ in range(width + (i < remainder))])
    return groups


def capacity_plan(topology, count, cores_per_replica=10):
    physical = set(topology["physical_core_cpus"])
    nodes = topology["numa_nodes"] or {0: topology["allowed_cpus"]}
    pools = {node: deque(cpu for cpu in cpus if cpu in physical) for node, cpus in nodes.items()}
    pools = {node: cpus for node, cpus in pools.items() if cpus}
    numa_counts = {node: dict(logical_cpus=len(cpus), physical_cores=len(pools.get(node, ())))
                   for node, cpus in nodes.items()}
    cpu_node = {cpu: node for node, cpus in pools.items() for cpu in cpus}
    if cores_per_replica is None:
        if count is None or not 1 <= count <= len(physical):
            raise ValueError(f"Full CPU assignment needs a replica count within 1–{len(physical)}")
        selected = distribute_all_cores(pools, count)
    else:
        groups = []
        while any(len(pool) >= cores_per_replica for pool in pools.values()):
            for pool in pools.values():
                if len(pool) >= cores_per_replica:
                    groups.append([pool.popleft() for _ in range(cores_per_replica)])
        # Pack complete extra replicas from residual CPUs across NUMA nodes.
        rest = [cpu for pool in pools.values() for cpu in pool]
        groups.extend(rest[i:i + cores_per_replica]
                      for i in range(0, len(rest) - cores_per_replica + 1, cores_per_replica))
        if not groups or (count is not None and count > len(groups)):
            raise ValueError(f"At most {len(groups)} replicas fit on {len(physical)} physical cores")
        selected = groups[:count]
    return dict(available_logical_cpus=len(topology["allowed_cpus"]),
                available_physical_cores=len(physical),
                numa_cpu_counts=numa_counts,
                cores_per_replica=cores_per_replica,
                assigned_physical_cores=sum(map(len, selected)),
                unused_cpu_ids=sorted(physical.difference(cpu for group in selected for cpu in group)),
                replicas=[dict(replica=i, cpus=cpus, nodes=sorted({cpu_node[cpu] for cpu in cpus}))
                          for i, cpus in enumerate(selected)])


def replica_error(out, rank, status):
    log = out / f"replica-{rank:02d}.log"
    if not log.is_file():
        return f"Replica {rank} exited with code {status}. Log: {log}"
    with log.open("rb") as file:
        file.seek(max(0, log.stat().st_size - 16384))
        tail = file.read().decode("utf-8", errors="replace").splitlines()[-50:]
    return f"Replica {rank} exited with code {status}. Last lines of {log}:\n" + "\n".join(tail)


def await_replicas(processes, gates, timeout, out, phase, completed=False):
    deadline = monotonic() + timeout
    while True:
        statuses = [proc.poll() for proc in processes]
        failed = [i for i, status in enumerate(statuses) if status is not None and status != 0]
        if failed:
            details = "\n\n".join(replica_error(out, i, statuses[i]) for i in failed[:3])
            raise RuntimeError(f"Replicas {failed} failed during {phase}.\n{details}")
        if completed:
            if all(status == 0 for status in statuses):
                return
        else:
            if all(path.is_file() for path in gates):
                return
            if any(status == 0 for status in statuses):
                raise RuntimeError(f"A replica exited before reporting {phase}. See logs in {out}")
        if monotonic() >= deadline:
            missing = [path.name for path in gates if not path.is_file()]
            pending = [i for i, status in enumerate(statuses) if status is None]
            raise TimeoutError(f"Timed out during {phase}. Active replicas: {pending}. "
                               f"Pending gates: {missing}. See logs in {out}")
        sleep(0.025)


def percentile_ms(values, fraction):
    values = sorted(values)
    return values[math.ceil(fraction * len(values)) - 1] / 1e6


def cpu_ticks(cpu_ids):
    ticks = {}
    for line in Path("/proc/stat").read_text().splitlines():
        name, *fields = line.split()
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        cpu = int(name[3:])
        if cpu in cpu_ids:
            values = list(map(int, fields[:8]))
            total = sum(values)
            ticks[cpu] = total - values[3] - values[4], total
    return ticks


def busy_fraction(before, after):
    busy = sum(after[cpu][0] - old[0] for cpu, old in before.items())
    total = sum(after[cpu][1] - old[1] for cpu, old in before.items())
    return busy / total if total > 0 else 0.0


def summarize_capacity(args, plan, summaries, cpu_busy):
    by_replica = []
    for replica, summary in zip(plan["replicas"], summaries, strict=True):
        hw = summary["hardware"]
        front_masks = hw["frontend_affinities"]
        if len(front_masks) != replica["frontend"]["api_server_count"]:
            raise RuntimeError("Missing observed frontend CPU masks")
        masks = [hw["engine_affinity"], *front_masks, *hw["worker_affinity"].values()]
        if replica["affinity"] == "compact":
            matches = sorted(sum(masks, [])) == sorted(replica["cpus"])
        else:
            matches = all(sorted(mask) == sorted(replica["cpus"]) for mask in masks)
        if not matches:
            raise RuntimeError(f"Replica {replica['replica']} did not honor its planned CPU set")
        if any(summary["parameters"][key] != value for key, value in replica["frontend"].items()):
            raise RuntimeError(f"Replica {replica['replica']} did not honor its frontend plan")
        samples = summary["workload"]["cold"]
        if len(samples) != args.runs:
            raise RuntimeError(f"Replica {replica['replica']} has incomplete measurements")
        by_replica.append(samples)
    if any((s["request_file_sha256"], s["metadata_sha256"], image_workload(s)) !=
           (summaries[0]["request_file_sha256"], summaries[0]["metadata_sha256"], image_workload(summaries[0]))
           for s in summaries):
        raise RuntimeError("Replicas did not receive the same input workload")
    rounds = []
    for number, samples in enumerate(zip(*by_replica, strict=True), 1):
        wall_ns = max(r["end_ns"] for r in samples) - min(r["start_ns"] for r in samples)
        seconds = wall_ns / 1e9
        req_count = len(samples) * args.concurrency_per_replica
        input_tokens = sum(r["input_tokens"] for r in samples)
        scheduled_prefill_tokens = sum(r["scheduled_prefill_tokens"] for r in samples)
        output_tokens = sum(r["output_tokens"] for r in samples)
        assert input_tokens == req_count * args.prompt_tokens
        assert output_tokens == req_count * args.new_tokens
        worker_cpu = sum(sum(r["worker_cpu_ns"]) for r in samples)
        front_cpu = sum(r["frontend_cpu_ns"] for r in samples)
        core_cpu = sum(r["engine_cpu_ns"] for r in samples)
        ttft = [t for r in samples for t in r["host_ttft_ns"]]
        latency = [t for r in samples for t in r["host_request_latency_ns"]]
        rounds.append(dict(
            round=number, requests=req_count, wall_s=seconds,
            start_skew_ms=(max(r["start_ns"] for r in samples) - min(r["start_ns"] for r in samples)) / 1e6,
            requests_per_s=req_count / seconds, input_tokens_per_s=input_tokens / seconds,
            scheduled_prefill_tokens_per_s=scheduled_prefill_tokens / seconds,
            output_tokens_per_s=output_tokens / seconds,
            host_ttft_p95_ms=percentile_ms(ttft, 0.95),
            host_latency_p95_ms=percentile_ms(latency, 0.95),
            frontend_cpu_cores=front_cpu / wall_ns, engine_cpu_cores=core_cpu / wall_ns,
            worker_cpu_cores=worker_cpu / wall_ns,
            effective_cpu_cores=(worker_cpu + front_cpu + core_cpu) / wall_ns,
            assigned_cpus_os_busy_fraction=cpu_busy[number - 1],
            renderer_peak_threads_per_frontend=max(r["renderer_peak_threads"] for r in samples),
            media_peak_threads_per_frontend=max(r["media_peak_threads"] for r in samples),
            replicas_with_image_text_overlap=sum(r["image_text_overlap"] for r in samples),
        ))
    return rounds


def image_workload(summary):
    image = summary["image"]
    if image is not None:
        return {key: value for key, value in image.items() if key != "media_loading_threads"}
    return None


def frontend_description(plan):
    counts = {}
    for replica in plan["replicas"]:
        if "frontend" not in replica:
            return "—"
        front = replica["frontend"]
        key = (front["api_server_count"], front["renderer_num_workers"], front["media_loading_threads"])
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{n} DP × {a}/{r}/{m}" for (a, r, m), n in sorted(counts.items()))


def compare_runs(paths):
    a, b = (json.loads(path.read_text()) for path in paths)
    if a.get("workload_profile") is None or b.get("workload_profile") is None:
        raise ValueError("Both summaries must contain a workload profile")
    if a["workload_profile"] != b["workload_profile"]:
        fields = a["workload_profile"].keys() | b["workload_profile"].keys()
        different = sorted(k for k in fields if a["workload_profile"].get(k) != b["workload_profile"].get(k))
        raise ValueError("Workload differs in " + ", ".join(different))
    print("| Metric | First host | Second host |")
    print("|---|---:|---:|")
    print(f"| Assigned physical cores | {a['cpu_plan']['assigned_physical_cores']} | "
          f"{b['cpu_plan']['assigned_physical_cores']} |")
    print(f"| Frontend processes / renderer threads / media threads | "
          f"{frontend_description(a['cpu_plan'])} | {frontend_description(b['cpu_plan'])} |")
    for label, key in (
        ("Requests/s", "requests_per_s"),
        ("Output tokens/s", "output_tokens_per_s"),
        ("Host TTFT P95 (ms)", "host_ttft_p95_ms"),
        ("Host latency P95 (ms)", "host_latency_p95_ms"),
        ("Effective CPU cores", "effective_cpu_cores"),
    ):
        print(f"| {label} | {a['capacity'][key]:.2f} | {b['capacity'][key]:.2f} |")


def reserve_output_dir(requested):
    requested = requested.absolute()
    requested = requested.parent.resolve() / requested.name
    if not requested.name:
        raise ValueError("Output directory needs a name")
    for suffix in count():
        path = requested.with_name(f"{requested.name}-{suffix}") if suffix else requested
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            # mkdir reserves the name atomically, including when several runs
            # start with the same requested output at the same time.
            continue
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--keep-diagnostics", action="store_true",
                        help="Keep per-round and per-DP data under the output directory")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("FIRST_JSON", "SECOND_JSON"),
                        help="Check two workload profiles and print comparable host metrics")
    parser.add_argument("--replicas", "--dp", type=int, default=8, help="Host DP replicas (default 8)")
    parser.add_argument("--tp", type=int, default=1, help="Logical workers per replica (default 1)")
    packing = parser.add_mutually_exclusive_group()
    packing.add_argument("--cores-per-replica", type=int,
                         help="Fixed physical cores per replica; TP + 2 pins each main role")
    packing.add_argument("--use-all-cores", action="store_true",
                         help="Spread all available physical cores over the replicas (default)")
    packing.add_argument("--core-sweep", nargs="*", type=int,
                         help=f"Measure total physical cores (default: {' '.join(map(str, STANDARD_CORE_SWEEP))})")
    parser.add_argument("--concurrency-per-replica", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--blocks", type=int, help="Override each DP logical KV block budget")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Maximum running sequences per DP")
    parser.add_argument("--chunk-tokens", type=int, default=16384)
    parser.add_argument("--request-file", type=Path)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--image-file", type=Path, help="One local JPEG or PNG per request")
    media.add_argument("--text-only", action="store_true", help="Skip the image codec stages")
    parser.add_argument("--api-server-count", type=int,
                        help="Total local frontend processes, divisible by DP (default: auto)")
    parser.add_argument("--renderer-num-workers", type=int,
                        help="Template and tokenizer threads per frontend (default: auto)")
    parser.add_argument("--media-loading-threads", type=int,
                        help="vLLM image input loading threads per frontend (default: auto)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--round-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.core_sweep == []:
        args.core_sweep = list(STANDARD_CORE_SWEEP)
    if args.compare:
        try:
            compare_runs(args.compare)
        except ValueError as exc:
            parser.error(str(exc))
        return
    if min(args.replicas, args.tp) < 1 or any(n is not None and n < 1 for n in
            (args.api_server_count, args.renderer_num_workers, args.media_loading_threads)):
        parser.error("DP, TP and frontend settings must be positive")
    if args.cores_per_replica is not None and args.cores_per_replica < 1:
        parser.error("--cores-per-replica must be positive")
    args.use_all_cores = args.cores_per_replica is None
    if args.blocks is not None and args.blocks < 1:
        parser.error("--blocks must be positive")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")
    if args.concurrency_per_replica < 1:
        parser.error("--concurrency-per-replica must be positive")
    api_per_replica = None
    if args.api_server_count is not None:
        api_per_replica, remainder = divmod(args.api_server_count, args.replicas)
        if remainder or not 1 <= api_per_replica <= args.concurrency_per_replica:
            parser.error("--api-server-count must be a multiple of DP, with one or more requests per frontend")
    if min(args.runs, args.timeout, args.round_timeout, args.prompt_tokens,
           args.new_tokens, args.chunk_tokens) <= 0 or args.warmup < 0:
        parser.error("Positive test counts and nonnegative warmup required")
    if args.prompt_tokens + args.new_tokens > 135000 or args.chunk_tokens % 128:
        parser.error("Total sequence length must fit 135000 and chunk size must align to 128")
    if args.core_sweep is not None and api_per_replica is not None:
        parser.error("The core sweep uses automatic frontend settings")
    if args.core_sweep is not None and (args.renderer_num_workers is not None or
                                      args.media_loading_threads is not None):
        parser.error("The core sweep uses automatic frontend settings")
    if args.core_sweep is not None and (len(set(args.core_sweep)) != len(args.core_sweep) or
                                      any(n < args.replicas or n % args.replicas for n in args.core_sweep)):
        parser.error("Core sweep totals must be distinct positive multiples of DP")

    topology = cpu_topology()
    try:
        if args.core_sweep is None:
            trials = [(args, make_plan(args, topology, api_per_replica))]
        else:
            trials = []
            for total in args.core_sweep:
                trial = argparse.Namespace(**(vars(args) | {
                    "cores_per_replica": total // args.replicas,
                    "use_all_cores": False,
                }))
                trials.append((trial, make_plan(trial, topology, api_per_replica)))
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_only:
        plans = [plan for _, plan in trials]
        print(json.dumps(plans if args.core_sweep else plans[0], indent=2))
        return
    if args.out is None:
        parser.error("--out is required for a measured run")
    if args.request_file is not None and not args.request_file.is_file():
        parser.error(f"Request file not found: {args.request_file}")

    root = Path(__file__).resolve().parent
    if not args.text_only:
        args.image_file = (args.image_file or root / "assets" / "input.jpg").resolve()
        if not args.image_file.is_file():
            parser.error(f"Image file not found: {args.image_file}")
    try:
        out = reserve_output_dir(args.out)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot create an output directory for {args.out}: {exc}")

    scratch = out / ".runs"
    scratch.mkdir()
    columns = []
    for trial, plan in trials:
        cores = plan["assigned_physical_cores"]
        trial.image_file = args.image_file
        trial.out = scratch / f"cores-{cores}"
        trial.out.mkdir()
        summaries = run_capacity(trial, plan, root)
        if args.keep_diagnostics:
            from stage_profile import write_profile
            write_profile(summaries, trial.out)
        heading = (f"{cores} 核（µs/请求）" if args.core_sweep else "摊销墙钟（µs/请求）")
        columns.append((heading, summaries))
    markdown = write_stage_tables(columns, out, diagnostics=args.keep_diagnostics)
    if not args.keep_diagnostics:
        shutil.rmtree(scratch)
    print(markdown)


def make_plan(args, topology, api_per_replica):
    plan = capacity_plan(topology, args.replicas, args.cores_per_replica)
    plan.update(dp_size=args.replicas, tp_size=args.tp, logical_cards=args.replicas * args.tp)
    for replica in plan["replicas"]:
        compact = args.cores_per_replica == args.tp + 2 and api_per_replica in (None, 1)
        replica["affinity"] = "compact" if compact else "inherit"
        replica["frontend"] = frontend_plan(
            len(replica["cpus"]), args.tp, args.concurrency_per_replica,
            not args.text_only, compact, api_per_replica,
            args.renderer_num_workers, args.media_loading_threads,
        )
    return plan


def run_capacity(args, plan, root):
    out = args.out
    (out / "cpu-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    gates = out / "round-gates"
    round_dirs = [gates / f"cold-{i}" for i in range(-args.warmup, args.runs)]
    reference_dirs = [gates / f"encode-cold-{i}" for i in range(args.runs)] if args.image_file else []
    for path in round_dirs + reference_dirs:
        path.mkdir(parents=True)
    processes = []
    cpu_busy = []
    assigned_cpus = {cpu for replica in plan["replicas"] for cpu in replica["cpus"]}
    try:
        for replica in plan["replicas"]:
            i = replica["replica"]
            directory = out / f"replica-{i:02d}"
            frontend = replica["frontend"]
            # Set the outer DP mask before Python imports or spawns any role or thread pool.
            cmd = ["taskset", "-c", ",".join(map(str, replica["cpus"])),
                   str(root / "run-native.sh"), "--cases", "cold", "--affinity", replica["affinity"],
                   "--tp", str(args.tp), "--concurrency", str(args.concurrency_per_replica),
                   "--api-server-count", str(frontend["api_server_count"]),
                   "--renderer-num-workers", str(frontend["renderer_num_workers"]),
                   "--prompt-tokens", str(args.prompt_tokens), "--new-tokens", str(args.new_tokens),
                   "--max-num-seqs", str(args.max_num_seqs),
                   "--chunk-tokens", str(args.chunk_tokens), "--warmup", str(args.warmup),
                   "--runs", str(args.runs), "--timeout", str(args.timeout),
                   "--round-timeout", str(args.round_timeout), "--round-gate", str(gates),
                   "--replica-id", str(i), "--out", str(directory)]
            if args.blocks is not None:
                cmd.extend(["--blocks", str(args.blocks)])
            if args.request_file:
                cmd.extend(["--request-file", str(args.request_file.resolve())])
            if args.image_file:
                cmd.extend(["--image-file", str(args.image_file), "--media-loading-threads",
                            str(frontend["media_loading_threads"])])
            with directory.with_suffix(".log").open("w") as log:
                processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                  env=os.environ | {"VLLM_HOST_PREFILL_RESOURCES": str(int(args.keep_diagnostics))},
                                                  start_new_session=True))
        for index, path in enumerate(round_dirs):
            await_replicas(processes, [path / f"ready-{i}" for i in range(len(processes))],
                           args.round_timeout, out, f"{path.name} readiness")
            before = cpu_ticks(assigned_cpus)
            (path / "start").touch()
            await_replicas(processes, [path / f"done-{i}" for i in range(len(processes))],
                           args.round_timeout, out, f"{path.name} measurement")
            if index >= args.warmup:
                cpu_busy.append(busy_fraction(before, cpu_ticks(assigned_cpus)))
        for path in reference_dirs:
            await_replicas(processes, [path / f"ready-{i}" for i in range(len(processes))],
                           args.round_timeout, out, "image encoding reference readiness")
            (path / "start").touch()
            await_replicas(processes, [path / f"done-{i}" for i in range(len(processes))],
                           args.round_timeout, out, "image encoding reference measurement")
        await_replicas(processes, [], args.round_timeout, out, "replica shutdown", completed=True)
    finally:
        pending = [proc for proc in processes if proc.poll() is None]
        for proc in pending:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in pending:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
    summaries = [json.loads((out / f"replica-{i:02d}" / "summary.json").read_text())
                 for i in range(len(processes))]
    rounds = summarize_capacity(args, plan, summaries, cpu_busy)
    with (out / "host-rounds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rounds[0]))
        writer.writeheader()
        writer.writerows(rounds)
    totals = {metric: median(r[metric] for r in rounds) for metric in rounds[0]
              if metric != "round"}
    totals["effective_cpu_fraction"] = totals["effective_cpu_cores"] / plan["assigned_physical_cores"]
    first = summaries[0]
    workload_profile = dict(
        replicas=len(summaries), request_file_sha256=first["request_file_sha256"],
        metadata_sha256=first["metadata_sha256"], versions=first["versions"], image=image_workload(first),
        frontend_pipeline="hf_renderer_rc5", round_end="last_serialized_response",
        **{key: first["parameters"][key] for key in (
            "tp", "concurrency", "prompt_tokens", "new_tokens", "chunk_tokens",
            "max_num_seqs", "max_model_len", "block_size", "blocks", "cases", "warmup", "runs",
        )},
    )
    (out / "host-summary.json").write_text(json.dumps(
        dict(capacity=totals, cpu_plan=plan, parameters=vars(args), per_round=rounds,
             workload_profile=workload_profile), default=str, indent=2) + "\n")
    return summaries


if __name__ == "__main__":
    main()
