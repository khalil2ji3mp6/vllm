"""Host subsets of the Ascend V1 runner. No device launch or speculative model."""
from functools import partial
from math import ceil
from time import perf_counter_ns, thread_time_ns

import numpy as np
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.block_table import BlockTable

from runner_report import MEASURED


class RunnerHost:
    def __init__(self, cfg):
        self.cfg = cfg
        max_reqs = cfg.scheduler_config.max_num_seqs
        width = cfg.model_config.max_model_len
        budget = cfg.scheduler_config.max_num_batched_tokens
        block_size = cfg.cache_config.block_size
        self.table = BlockTable(
            block_size=block_size, max_num_reqs=max_reqs, max_num_blocks_per_req=ceil(width / block_size),
            max_num_batched_tokens=budget, pin_memory=False, device=torch.device("cpu"),
            kernel_block_size=block_size, cp_kv_cache_interleave_size=1,
        )
        self.token_rows = torch.empty((max_reqs, width), dtype=torch.int32)
        self.token_np = self.token_rows.numpy()
        self.ids = torch.empty(budget, dtype=torch.int32)
        self.arange = np.arange(max(max_reqs, budget), dtype=np.int32)
        self.positions = np.empty(budget, dtype=np.int64)
        self.query_pos = np.empty(budget, dtype=np.int32)
        self.computed = np.empty(max_reqs, dtype=np.int32)
        self.row_states = [None] * max_reqs

    def reset(self):
        # On the next scheduled step each occupied row will be initialized afresh.
        self.row_states[:] = [None] * len(self.row_states)

    def update(self, batch, output):
        batch.update(output)
        cached = output.scheduled_cached_reqs
        for row, (rid, n) in enumerate(output.num_scheduled_tokens.items()):
            state = batch.reqs[rid]
            end = state.computed + n
            if len(state.blocks) * batch.block_size < end:
                raise ValueError(f"Missing logical KV blocks for {rid}")
            if self.row_states[row] is not state or rid in cached.resumed_req_ids:
                self.table.add_row(state.blocks, row)
                self.token_np[row, :len(state.prompt)] = state.prompt
                self.row_states[row] = state
            else:
                used = self.table.num_blocks_per_row[row]
                if used < len(state.blocks):
                    self.table.append_row(state.blocks[used:], row)
            if end > len(state.prompt):
                self.token_np[row, max(state.computed, len(state.prompt)):end] = batch.token_id
            self.computed[row] = state.computed

    def prepare(self, output):
        counts = np.fromiter(output.num_scheduled_tokens.values(), dtype=np.int32,
                             count=len(output.num_scheduled_tokens))
        size = output.total_num_scheduled_tokens
        n = len(counts)
        rows = np.repeat(self.arange[:n], counts)
        ends = np.cumsum(counts, dtype=np.int32)
        np.subtract(self.arange[:size], np.repeat(ends - counts, counts), out=self.query_pos[:size])
        positions = self.positions[:size]
        np.add(self.computed[rows], self.query_pos[:size], out=positions)
        indices = torch.from_numpy(positions + rows * self.token_np.shape[1])
        ids = self.ids[:size]
        torch.index_select(self.token_rows.view(-1), 0, indices, out=ids)
        starts = np.empty(n + 1, dtype=np.int32)
        starts[0], starts[1:] = 0, ends
        lengths = torch.from_numpy(self.computed[:n] + counts)
        tensors = (ids, torch.from_numpy(positions), lengths, torch.from_numpy(starts),
                   self.table.get_cpu_tensor()[:n])
        # The Ascend backend computes slot values with a device kernel. Only
        # pass the buffer view through the host metadata container here.
        return tensors, self.table.slot_mapping.cpu[:size], int(counts.max())

    def padding(self, output, inputs):
        tensors, _, max_query = inputs
        n = len(tensors[2])
        has_state = bool(np.all(self.computed[:n] > 0))
        total = output.total_num_scheduled_tokens
        uniform = has_state and max_query == 1 and total == n
        # TP1, eager: the dispatch result keeps the actual token count.
        return BatchDescriptor(total), uniform

    @staticmethod
    def metadata(inputs, shape):
        tensors, slots, max_query = inputs
        _, positions, lengths, starts, blocks = tensors
        descriptor, _ = shape
        return CommonAttentionMetadata(
            query_start_loc=starts, query_start_loc_cpu=starts, seq_lens=lengths,
            num_reqs=len(lengths), num_actual_tokens=descriptor.num_tokens,
            max_query_len=max_query, max_seq_len=int(lengths.numpy().max()),
            block_table_tensor=blocks, slot_mapping=slots, positions=positions,
        )

    @staticmethod
    def preprocess(inputs, shape):
        tensors = inputs[0]
        size = shape[0].num_tokens
        # Text IDs and 1-D positions. Images are decoded in the frontend only.
        return tensors[0][:size], tensors[1][:size]

    def context(self, metadata, shape):
        desc = shape[0]
        # Common vLLM enter + exit, with no forward body or Ascend extension.
        with set_current_vllm_config(self.cfg), set_forward_context(
            metadata, self.cfg, num_tokens=desc.num_tokens, batch_descriptor=desc
        ):
            pass

    def step(self, batch, output, clock, phase):
        def call(key, func, *args):
            cpu = thread_time_ns()
            start = perf_counter_ns()
            value = func(*args)
            end = perf_counter_ns()
            clock.rows[phase, key].append((start, end, thread_time_ns() - cpu))
            return value

        update, prepare, padding, attn, preprocess, context = MEASURED
        call(update, self.update, batch, output)
        inputs = call(prepare, self.prepare, output)
        shape = call(padding, self.padding, output, inputs)
        metadata = call(attn, self.metadata, inputs, shape)
        forward_inputs = call(preprocess, self.preprocess, inputs, shape)
        call(context, self.context, metadata, shape)
        return forward_inputs

    def isolated(self, batch, output):
        """Freeze prerequisite values so each measured call runs one operation."""
        self.update(batch, output)
        inputs = self.prepare(output)
        shape = self.padding(output, inputs)
        metadata = self.metadata(inputs, shape)
        operations = (
            partial(self.update, batch, output), partial(self.prepare, output),
            partial(self.padding, output, inputs), partial(self.metadata, inputs, shape),
            partial(self.preprocess, inputs, shape), partial(self.context, metadata, shape),
        )
        return dict(zip(MEASURED, operations, strict=True))
