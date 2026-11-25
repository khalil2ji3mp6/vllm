#!/usr/bin/env python3
"""Run a fixed integer workload on one CPU and on the full Linux CPU mask."""
import argparse
import json
import math
import multiprocessing as mp
import os
import platform
from pathlib import Path
from statistics import median
from time import monotonic_ns, process_time_ns, sleep

from cpu_topology import cpu_topology


PRIME_LIMIT = 10_000
EXPECTED_PRIMES = 1_229


def count_primes():
    count = 1
    for candidate in range(3, PRIME_LIMIT + 1, 2):
        for divisor in range(3, math.isqrt(candidate) + 1, 2):
            if candidate % divisor == 0:
                break
        else:
            count += 1
    return count


def stress_worker(cpu, duration_ns, gate, start_ns, result):
    os.sched_setaffinity(0, {cpu})
    if count_primes() != EXPECTED_PRIMES:
        raise RuntimeError("Unexpected prime count")
    gate.wait(timeout=60)
    scheduled = start_ns.value
    remaining_ns = scheduled - monotonic_ns()
    if remaining_ns > 0:
        sleep(remaining_ns / 1e9)

    started = monotonic_ns()
    deadline = scheduled + duration_ns
    cpu_before = process_time_ns()
    jobs = 0
    finished = started
    while finished < deadline:
        count_primes()
        finished = monotonic_ns()
        jobs += finished <= deadline
    result[:] = (jobs, process_time_ns() - cpu_before, started, finished)


def measure(cpus, seconds):
    ctx = mp.get_context("fork")
    start_ns = ctx.RawValue("q", 0)
    duration_ns = int(seconds * 1e9)

    def set_start():
        start_ns.value = monotonic_ns() + 100_000_000

    gate = ctx.Barrier(len(cpus) + 1, action=set_start)
    results = [ctx.RawArray("q", 4) for _ in cpus]
    processes = []
    try:
        for cpu, result in zip(cpus, results, strict=True):
            child = ctx.Process(target=stress_worker, args=(cpu, duration_ns, gate, start_ns, result))
            child.start()
            processes.append(child)
        gate.wait(timeout=60)
        deadline = start_ns.value + duration_ns + 30_000_000_000
        for cpu, child in zip(cpus, processes, strict=True):
            child.join(max(0, deadline - monotonic_ns()) / 1e9)
            if child.exitcode != 0:
                raise RuntimeError(f"CPU {cpu} worker did not complete (exit code {child.exitcode})")
    finally:
        for child in processes:
            if child.is_alive():
                child.terminate()
            child.join()

    begin = start_ns.value
    active_ns = max(begin + duration_ns, *(result[3] for result in results)) - begin
    jobs = sum(result[0] for result in results)
    effective_cpus = sum(result[1] for result in results) / active_ns
    return {
        "workers": len(cpus),
        "jobs": jobs,
        "jobs_per_second": jobs / seconds,
        "effective_cpus": effective_cpus,
        "worker_utilization_pct": 100 * effective_cpus / len(cpus),
        "start_skew_ms": (max(row[2] for row in results) - min(row[2] for row in results)) / 1e6,
        "per_cpu": [
            {"cpu": cpu, "jobs": result[0], "cpu_ms": result[1] / 1e6}
            for cpu, result in zip(cpus, results, strict=True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5, help="Seconds per run (default: 5)")
    parser.add_argument("--logical", action="store_true", help="Use all logical CPUs, including SMT siblings")
    parser.add_argument("--out", type=Path, help="Save results as JSON")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("This benchmark uses Linux CPU affinity and sysfs topology")
    if not math.isfinite(args.seconds) or args.seconds < 0.1:
        parser.error("--seconds must be a finite number of at least 0.1")

    topology = cpu_topology()
    selected = topology["allowed_cpus" if args.logical else "physical_core_cpus"]
    baseline_cpus = sorted({selected[0], selected[len(selected) // 2], selected[-1]})
    single_samples = [measure([cpu], args.seconds) for cpu in baseline_cpus]
    full = measure(selected, args.seconds) if len(selected) > 1 else single_samples[0]
    if any(run["jobs"] == 0 for run in (*single_samples, full)):
        parser.error("No work completed inside the measurement window. Increase --seconds and retry.")
    single = {"workers": 1}
    for key in ("jobs_per_second", "effective_cpus", "worker_utilization_pct"):
        single[key] = median(run[key] for run in single_samples)
    speedup = full["jobs_per_second"] / single["jobs_per_second"]
    results = {
        "workload": f"CPython integer prime count through {PRIME_LIMIT}",
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "seconds": args.seconds,
        "cpu_selection": "logical" if args.logical else "physical",
        "allowed_logical_cpus": len(topology["allowed_cpus"]),
        "available_physical_cores": len(topology["physical_core_cpus"]),
        "single": single,
        "single_samples": single_samples,
        "full": full,
        "speedup": speedup,
    }
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    print("| 场景 | 进程数 | 质数搜索轮次/秒 | 有效 CPU 核数 | 进程 CPU 占用率 |")
    print("|---|---:|---:|---:|---:|")
    for name, run in (("单核", single), ("全核", full)):
        print(f"| {name} | {run['workers']} | {run['jobs_per_second']:,.2f} | "
              f"{run['effective_cpus']:.2f} | {run['worker_utilization_pct']:.1f}% |")
    sample_rates = [run["jobs_per_second"] for run in single_samples]
    print(f"单核采样范围 {min(sample_rates):,.2f}–{max(sample_rates):,.2f} 轮/秒，"
          f"全核吞吐为单核中位数的 {speedup:.2f}×")


if __name__ == "__main__":
    main()
