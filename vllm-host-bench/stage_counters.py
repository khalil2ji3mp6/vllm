"""Optional Linux userspace PMU counts for the calling lane; no perf dependency."""

import ctypes
import os
import platform
import struct


class Counters:
    def __init__(self):
        self.fds = {}
        self.errors = {}
        number = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
        if number is None:
            return
        syscall = ctypes.CDLL(None, use_errno=True).syscall
        # perf_event_attr version 0. User mode only; scale multiplexed counts
        # using PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING.
        for name, event in (("cycles", 0), ("instructions", 1), ("cache_misses", 3)):
            attr = ctypes.create_string_buffer(64)
            struct.pack_into(
                "=IIQQQQQ", attr, 0, 0, 64, event, 0, 0, 3, (1 << 5) | (1 << 6)
            )
            fd = syscall(number, ctypes.byref(attr), 0, -1, -1, 0)
            if fd < 0:
                self.errors[name] = os.strerror(ctypes.get_errno())
            else:
                self.fds[name] = fd

    def read(self):
        samples = {}
        for name, fd in self.fds.items():
            try:
                samples[name] = struct.unpack("=QQQ", os.read(fd, 24))
            except (OSError, struct.error) as exc:
                self.errors[name] = str(exc)
        return samples

    def deltas(self, previous):
        result = {}
        for name, (value, enabled, running) in self.read().items():
            if name not in previous:
                continue
            start, on, active = previous[name]
            if running > active:
                result[name] = (value - start) * (enabled - on) / (running - active)
        return result

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
