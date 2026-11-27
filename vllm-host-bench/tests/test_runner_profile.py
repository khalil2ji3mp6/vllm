"""Regression checks for persistent rows and pooled exclusive timing."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import bench  # noqa: F401 -- select the same CPU-only platform as run-stages.sh
import torch

from runner_host import RunnerHost
from runner_report import MEASURED, SPEC_ROWS, collect_workers, write_tables
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

    def test_speculative_host_boundaries(self):
        from engine import config
        from runner_spec_host import SpecHostCases
        from vllm.config import CUDAGraphMode

        args = NS(model_metadata=Path(__file__).resolve().parents[1] / "model",
                  max_model_len=135000, renderer_num_workers=1, block_size=128,
                  max_num_seqs=32, chunk_tokens=16384, tp=1, blocks=8192)
        cfg = config(args, skip_tokenizer_init=True)
        original_mode = cfg.compilation_config.cudagraph_mode
        sample = SpecHostCases(cfg)
        self.assertEqual(cfg.compilation_config.cudagraph_mode, original_mode)
        self.assertIs(sample.model_forward().keywords["positions"], sample.positions)
        self.assertEqual(sample.compute_logits().data_ptr(), sample.logits.data_ptr())
        self.assertEqual(sample.parse_verified_ids(), ([[11, 17]], None))
        bonus, output = sample.rej_sample()
        self.assertEqual(bonus.max_num_logprobs, -1)
        self.assertIsNone(sample.sampling.max_num_logprobs)
        self.assertIs(output.sampled_token_ids, sample.verified)
        mode, descriptor = sample.draft_padding()
        self.assertEqual(mode, CUDAGraphMode.PIECEWISE)
        self.assertEqual(descriptor.num_tokens, 4)
        self.assertEqual([t.tolist() for t in sample.propose()],
                         [[17], [32770], [0, 2], [0, 1]])

    def test_two_tables_with_shared_decode_denominator(self):
        # Every regular row averages 1 us. Each speculative row averages 2 us
        # with a different sample count. Bookkeeping must not affect the sum.
        ordinary = {k: {"calls": 2, "wall_sum_ns": 2000} for k in MEASURED}
        spec = {k: {"calls": 3, "wall_sum_ns": 6000} for _, _, k in SPEC_ROWS}
        phases = dict(prefill=ordinary, decode=ordinary, speculative=spec,
                      bookkeeping={"parse_verified_ids": {"calls": 1, "wall_sum_ns": 999999}})
        with TemporaryDirectory() as directory:
            report = write_tables({1: phases, 32: phases}, Path(directory))
            self.assertEqual([s for s in report.splitlines() if s.startswith("## ")],
                             ["## Prefill", "## Decode"])
            self.assertIn("32 核占比", report)
            self.assertIn("| propose | propose | 跨 DP 并行，步内同步 | 2.00 | 12.50% | 2.00 | 12.50% |", report)
            self.assertIn("| **16.00** | 100%", report)
            whole = write_tables({32: dict(prefill=ordinary, decode={})}, Path(directory))
            self.assertIn("| propose | propose | — | — | — |", whole)

    def test_renderer_uses_service_roles_and_keeps_worker_sum(self):
        from renderer_bench import service_plan
        cpus = list(range(192))
        topo = dict(allowed_cpus=cpus, physical_core_cpus=cpus,
                    numa_nodes={i: cpus[i*48:(i+1)*48] for i in range(4)})
        item = {"calls": 2, "wall_sum_ns": 4000, "ops_s": 123.4}
        profiles = {}
        for count, dp, apis, threads in ((1, 1, 1, 1), (32, 8, 1, 4),
                                         (64, 8, 1, 8), (192, 8, 3, 8)):
            plan = service_plan(topo, count)
            self.assertEqual(len(plan["replicas"]), dp)
            self.assertEqual(len({cpu for r in plan["replicas"] for cpu in r["cpus"]}), count)
            for r in plan["replicas"]:
                self.assertEqual((r["frontend"]["api_server_count"],
                                  r["frontend"]["renderer_num_workers"]), (apis, threads))
            profiles[count] = dict(service_plan=plan, prefill={MEASURED[0]: item}, decode={},
                                   frontend_tokenizer=dict(calls=100, wall_sum_ns=1800000, ops_s=300,
                                                           peak_by_api=[threads] * dp * apis))
        uneven = service_plan(topo, 33)
        self.assertEqual((uneven["dp_size"], uneven["assigned_physical_cores"]), (8, 33))
        with TemporaryDirectory() as directory:
            md = write_tables(profiles, Path(directory))
            self.assertIn("Tokenizer | 跨请求多核，renderer 线程池 | 18.00 | — | 300.00", md)
            self.assertIn("_update_states | 跨 DP 并行，步内同步 | 2.00 | 100.00% | 123.40", md)
            self.assertIn("192 核 8 DP × 3 API/DP × 8 线程/API", md)

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
