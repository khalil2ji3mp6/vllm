"""Regression checks for persistent rows and pooled exclusive timing."""
import unittest
from types import SimpleNamespace as NS

import torch

from runner_host import RunnerHost
from runner_report import MEASURED, collect_workers
from worker import CpuBatch


def output(counts, new=(), cached=(), finished=()):
    return NS(
        num_scheduled_tokens=dict(counts), total_num_scheduled_tokens=sum(n for _, n in counts),
        scheduled_new_reqs=[NS(req_id=rid, prompt_token_ids=ids, num_computed_tokens=0,
                               block_ids=(blocks,)) for rid, ids, blocks in new],
        finished_req_ids=finished,
        scheduled_cached_reqs=NS(
            req_ids=[rid for rid, _, _ in cached], num_computed_tokens=[n for _, n, _ in cached],
            new_block_ids=[(delta,) for _, _, delta in cached], resumed_req_ids=set(),
        ),
    )


class RunnerChecks(unittest.TestCase):
    def test_step_tensors_after_reorder_append_and_reuse(self):
        cfg = NS(model_config=NS(max_model_len=16), scheduler_config=NS(
            max_num_seqs=2, max_num_batched_tokens=8), cache_config=NS(block_size=2))
        host = RunnerHost(cfg)
        state, reference = CpuBatch(2, 99), CpuBatch(2, 99)
        steps = [
            output([("a", 2), ("b", 2)], new=[("a", [20, 21], [1]), ("b", [30, 31], [3])]),
            output([("b", 1), ("a", 1)], cached=[("a", 2, [2]), ("b", 2, [4])]),
            output([("b", 1), ("a", 1)], cached=[("b", 3, []), ("a", 3, [])]),
            output([("b", 1), ("a", 1)], cached=[("b", 4, [5]), ("a", 4, [6])]),
            output([("b", 2)], new=[("b", [40, 41], [7])], finished=["b", "a"]),
        ]
        for step in steps:
            host.update(state, step)
            reference.update(step)
            tensors, slots, max_query = host.prepare(step)
            expected = reference.prepare(step)
            for actual, original in zip(tensors[:4], expected[:4], strict=True):
                torch.testing.assert_close(actual, original, check_dtype=False)
            for row, rid in enumerate(step.num_scheduled_tokens):
                width = len(reference.reqs[rid].blocks)
                self.assertEqual(tensors[4][row, :width].tolist(), reference.reqs[rid].blocks)
            self.assertEqual(slots.numel(), step.total_num_scheduled_tokens)
            self.assertEqual(max_query, max(step.num_scheduled_tokens.values()))

    def test_mixed_counted_once_and_dropped_span_rejected(self):
        summary = {"parameters": {"tp": 1}, "workload": {"cold": [{"mixed_steps": 2, "decode_steps": 1}]}}
        summary["timings"] = [
            dict(case="cold", process="worker0", phase=phase, stage=key, calls=n, wall_sum_ns=n*100)
            for phase, n in (("mixed", 2), ("prefill_inclusive", 2), ("decode", 1)) for key in MEASURED
        ]
        result = collect_workers([summary, summary])
        self.assertEqual(result["prefill"][MEASURED[0]], {"calls": 4, "wall_sum_ns": 400})
        self.assertEqual(result["decode"][MEASURED[0]]["calls"], 2)
        summary["timings"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete decode"):
            collect_workers([summary])


if __name__ == "__main__":
    unittest.main()
