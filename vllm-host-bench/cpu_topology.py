"""Linux CPU affinity, NUMA nodes and physical-core representatives."""
import os
from pathlib import Path


def cpulist(text):
    cpus = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpu_topology(numa_node=None):
    allowed = set(os.sched_getaffinity(0))
    node_cpus = {
        int(file.parent.name[4:]): sorted(allowed.intersection(cpulist(file.read_text().strip())))
        for file in Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")
    }
    node_cpus = {node: node_cpus[node] for node in sorted(node_cpus) if node_cpus[node]}
    if numa_node is not None:
        if numa_node not in node_cpus:
            raise ValueError(f"NUMA node {numa_node} has no CPUs in the current affinity mask")
        node_cpus = {numa_node: node_cpus[numa_node]}
        allowed = set(node_cpus[numa_node])
    nodes = list(node_cpus.values()) or [sorted(allowed)]
    # Choose one allowed logical CPU per physical core, including when a
    # taskset/cpuset exposes only the second SMT sibling.
    physical = {}
    for cpu in sorted(allowed):
        file = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        siblings = tuple(cpulist(file.read_text().strip())) if file.is_file() else (cpu,)
        physical.setdefault(siblings, cpu)
    representatives = set(physical.values())
    return dict(allowed_cpus=sorted(allowed), numa_cpus=nodes, numa_nodes=node_cpus,
                physical_core_cpus=sorted(representatives))
