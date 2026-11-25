"""Per-occurrence wall and calling-thread CPU spans."""
import os
import resource
from collections import Counter, defaultdict
from contextlib import contextmanager
from statistics import median
from time import perf_counter_ns, thread_time_ns


PREFILL_RESOURCES = os.environ.get("VLLM_HOST_PREFILL_RESOURCES") == "1"


class Clock:
    def __init__(self):
        self.rows = defaultdict(list)
        self.resources = defaultdict(Counter) if PREFILL_RESOURCES else None

    @contextmanager
    def span(self, group, name):
        start_wall = perf_counter_ns()
        start_cpu = thread_time_ns()
        try:
            yield
        finally:
            cpu = thread_time_ns() - start_cpu
            self.rows[group, name].append((start_wall, perf_counter_ns(), cpu))

    def prefill(self, name):
        if self.resources is None:
            return self.span("prefill", name)
        return self.prefill_resources(name)

    @contextmanager
    def prefill_resources(self, name):
        before = resource.getrusage(resource.RUSAGE_THREAD)
        try:
            with self.span("prefill", name):
                yield
        finally:
            after = resource.getrusage(resource.RUSAGE_THREAD)
            self.resources[name].update({
                "calls": 1,
                "user_us": (after.ru_utime - before.ru_utime) * 1e6,
                "system_us": (after.ru_stime - before.ru_stime) * 1e6,
                "minor_faults": after.ru_minflt - before.ru_minflt,
                "major_faults": after.ru_majflt - before.ru_majflt,
                "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
            })

    def merge_resources(self, other):
        if other.resources:
            for name, counts in other.resources.items():
                self.resources[name].update(counts)

    def move_last(self, name, group):
        self.rows[group, name].append(self.rows["pending", name].pop())


def distribution(samples):
    def quantile(values):
        values = sorted(values)
        return [
            round(median(values) / 1e3, 3),
            round(values[(95 * len(values) + 99) // 100 - 1] / 1e3, 3),
            sum(values),
        ]

    wall = quantile([end - start for start, end, _ in samples])
    cpu = quantile([cpu for _, _, cpu in samples])
    return dict(calls=len(samples), wall_p50_us=wall[0], wall_p95_us=wall[1],
                thread_cpu_p50_us=cpu[0], thread_cpu_p95_us=cpu[1],
                wall_sum_ns=wall[2], thread_cpu_sum_ns=cpu[2])


def summarize(clock):
    return [
        {"phase": phase, "stage": stage, **distribution(samples)}
        for (phase, stage), samples in sorted(clock.rows.items()) if samples
    ]
