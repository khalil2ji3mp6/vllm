"""Per-occurrence wall and calling-thread CPU spans."""
from collections import defaultdict
from contextlib import contextmanager
from statistics import median
from time import perf_counter_ns, thread_time_ns


class Clock:
    def __init__(self):
        self.rows = defaultdict(list)

    @contextmanager
    def span(self, group, name):
        start_wall = perf_counter_ns()
        start_cpu = thread_time_ns()
        try:
            yield
        finally:
            cpu = thread_time_ns() - start_cpu
            self.rows[group, name].append((start_wall, perf_counter_ns(), cpu))

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
