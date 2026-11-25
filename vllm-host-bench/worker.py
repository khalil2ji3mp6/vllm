"""CPU-only runner boundary. Prepare descriptors; never run a model or launch ops."""
import os
from time import perf_counter_ns, process_time_ns
from types import SimpleNamespace

import numpy as np
import torch
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

from engine import dummy_result
from measure import Clock


class CpuBatch:
    def __init__(self, block_size, token_id):
        self.block_size = block_size
        self.token_id = token_id
        self.reqs = {}
        self.positions_base = np.arange(16384, dtype=np.int32)

    def update(self, output):
        for rid in output.finished_req_ids:
            self.reqs.pop(rid, None)
        for req in output.scheduled_new_reqs:
            self.reqs[req.req_id] = SimpleNamespace(
                prompt=np.asarray(req.prompt_token_ids, dtype=np.int32),
                computed=req.num_computed_tokens,
                blocks=list(req.block_ids[0]),
            )
        cached = output.scheduled_cached_reqs
        for rid, computed, block_delta in zip(
            cached.req_ids, cached.num_computed_tokens, cached.new_block_ids, strict=True,
        ):
            state = self.reqs[rid]
            state.computed = computed
            if block_delta:
                if rid in cached.resumed_req_ids:
                    state.blocks = list(block_delta[0])
                else:
                    state.blocks.extend(block_delta[0])

    def prepare(self, output):
        scheduled = output.num_scheduled_tokens
        total = output.total_num_scheduled_tokens
        counts = np.fromiter(scheduled.values(), dtype=np.int32, count=len(scheduled))
        seq_lens = np.empty(len(scheduled), dtype=np.int32)
        query_start = np.empty(len(scheduled) + 1, dtype=np.int32)
        query_start[0] = 0
        np.cumsum(counts, out=query_start[1:])
        ids = np.empty(total, dtype=np.int32)
        positions = np.empty(total, dtype=np.int32)
        rows = [self.reqs[rid].blocks for rid in scheduled]
        block_table = np.zeros((len(rows), max(map(len, rows), default=0)), dtype=np.int32)
        for row, (rid, count) in enumerate(scheduled.items()):
            req = self.reqs[rid]
            first = req.computed
            end = first + count
            lo, hi = query_start[row], query_start[row + 1]
            prefix_end = min(end, len(req.prompt))
            n_prompt = max(0, prefix_end - first)
            ids[lo:lo + n_prompt] = req.prompt[first:prefix_end]
            ids[lo + n_prompt:hi] = self.token_id
            if count <= len(self.positions_base):
                positions[lo:hi] = self.positions_base[:count] + first
            else:
                positions[lo:hi] = np.arange(first, end, dtype=np.int32)
            seq_lens[row] = end
            block_table[row, :len(req.blocks)] = req.blocks
            if req.blocks and len(req.blocks) * self.block_size < end:
                raise RuntimeError(f"Insufficient logical block table for {rid}")
        return tuple(torch.from_numpy(x) for x in (ids, positions, seq_lens, query_start, block_table))


def run_worker(args, rank, handle, control, token_id):
    torch.set_num_threads(1)
    rx = MessageQueue.create_from_handle(handle, rank)
    reply = MessageQueue(1, 1, max_chunk_bytes=1 << 20, max_chunks=2) if rank == 0 else None
    control.send({
        "reply": reply.export_handle() if reply else None,
        "affinity": sorted(os.sched_getaffinity(0)),
    })
    rx.wait_until_ready()
    if reply:
        reply.wait_until_ready()

    batch = CpuBatch(args.block_size, token_id)
    clock = Clock()
    while True:
        # Waiting for work is represented by end-to-end dispatch/response wall spans.
        with clock.span("pending", "worker_dequeue_wait_and_deserialize"):
            item = rx.dequeue(timeout=args.timeout)
        if item[0] == "stop":
            break
        if item[0] == "begin":
            cpu_start = process_time_ns()
            batch = CpuBatch(args.block_size, token_id)
            clock = Clock()
            continue
        if item[0] == "end":
            control.send((dict(clock.rows), process_time_ns() - cpu_start, perf_counter_ns()))
            continue
        phase, output, sample = item
        clock.move_last("worker_dequeue_wait_and_deserialize", phase)
        with clock.span(phase, "worker_prepare_cpu"):
            with clock.span(phase, "worker_batch_update"):
                batch.update(output)
            with clock.span(phase, "worker_cpu_metadata"):
                batch.prepare(output)
        if reply:
            with clock.span(phase, "fixture_token_output"):
                result = dummy_result(sample, token_id)
            with clock.span(phase, "worker_result_enqueue"):
                reply.enqueue(result, timeout=args.timeout)
    # This synchronous reader has already received the stop message. Close
    # its private ZeroMQ context, including notify and cancel sockets, with
    # no pending writes to drain.
    for queue in (rx, reply):
        if queue is not None:
            queue.local_socket.context.destroy(linger=0)
