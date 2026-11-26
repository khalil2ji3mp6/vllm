"""Read-only Linux placement observations and optional child-local MPOL_BIND."""

import ctypes as C
import os
from collections import Counter
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")
try:
    NUMA = C.CDLL("libnuma.so.1", use_errno=True)
except OSError:
    NUMA = None


def bind_memory(node):
    if NUMA is None:
        return "libnuma.so.1 unavailable"
    bits = C.sizeof(C.c_ulong) * 8
    # Give the kernel a word-sized nodemask with a spare high bit, including
    # when the requested node is the final bit of a word.
    mask = (C.c_ulong * ((node + 1) // bits + 1))()
    mask[node // bits] = 1 << (node % bits)
    if NUMA.set_mempolicy(2, mask, C.c_ulong(len(mask) * bits)) != 0:
        return os.strerror(C.get_errno())
    return None


def sample_addresses(array):
    first = array.ctypes.data // PAGE
    last = (array.ctypes.data + array.nbytes - 1) // PAGE
    n = min(64, last - first + 1)
    return [(first + i * (last - first) // max(n - 1, 1)) * PAGE for i in range(n)]


def vma_nodes(addresses):
    # Only attribute a sampled address when its entire resident VMA is on one node.
    # Malloc can merge our buffers with unrelated ranges. Mixed VMAs stay unknown.
    policies = {}
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        start, *fields = line.split()
        nodes = [f.split("=")[0][1:] for f in fields if f.startswith("N") and "=" in f]
        policies[int(start, 16)] = nodes[0] if len(nodes) == 1 else "?"
    ranges = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        start, end = (int(x, 16) for x in line.split()[0].split("-"))
        if any(start <= a < end for a in addresses):
            ranges.append((start, end, policies.get(start, "?")))
    return Counter(
        next((node for a, b, node in ranges if a <= p < b), "?") for p in addresses
    )


def placement(array):
    addresses = sample_addresses(array)
    reason = "libnuma.so.1 unavailable"
    if NUMA is not None:
        pages = (C.c_void_p * len(addresses))(*addresses)
        status = (C.c_int * len(addresses))(*([-1] * len(addresses)))
        if NUMA.move_pages(0, C.c_ulong(len(pages)), pages, None, status, 0) >= 0:
            return {
                "method": "move_pages",
                "nodes": dict(Counter(str(n) if n >= 0 else "?" for n in status)),
            }
        reason = os.strerror(C.get_errno())
    try:
        nodes = vma_nodes(addresses)
    except OSError as exc:
        nodes = {"?": len(addresses)}
        reason += f"; procfs: {exc}"
    return {"method": "numa_maps", "reason": reason, "nodes": dict(nodes)}
