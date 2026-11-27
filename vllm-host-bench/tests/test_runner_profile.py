"""Regression checks for persistent rows and pooled exclusive timing."""
import unittest
from unittest.mock import patch
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

    def test_worker_fixture_skips_tokenizer(self):
        from stage_cases import Cases
        with patch("stage_cases.GlmRc5Tokenizer", side_effect=AssertionError("unexpected tokenizer")):
            fixture = Cases(["runner/prefill/worker_batch_update"], 1)
        self.assertEqual(len(fixture.requests[0].prompt_token_ids), 32768)
        self.assertEqual(fixture.ids_hash.hexdigest(),
                         "edb6bccbc6d08dc652681680f6705adb5675ec579e948083c79919f694e01662")
        item = dict(calls=2, wall_sum_ns=4000)
        with TemporaryDirectory() as directory:
            md = write_tables({1: dict(dp_size=1, tp_size=1, prefill={MEASURED[0]: item},
                                       decode={})}, Path(directory))
        self.assertIn("_update_states | 2.00 | 100.00%", md)
        self.assertNotIn("Tokenizer", md)
        self.assertNotIn("多核配置", md)

    def test_hygon_core_sweep_respects_memory_node_affinity(self):
        from cpu_topology import benchmark_topology, hygon_pool
        from host_capacity import make_plan

        nodes = {i: list(range(i*16, (i+1)*16)) for i in range(8)}
        cpus = list(range(128))
        full = dict(physical_core_cpus=cpus, allowed_cpus=cpus, numa_nodes=nodes)
        selected = nodes[2] + nodes[6]
        scoped = dict(physical_core_cpus=selected, allowed_cpus=selected,
                      numa_nodes={i: nodes[i] for i in (2, 6)})
        with (patch("cpu_topology.cpu_topology", side_effect=[full, scoped]),
              patch("cpu_topology.platform.machine", return_value="x86_64"),
              patch("cpu_topology.Path.read_text", return_value="vendor_id: HygonGenuine"),
              patch("cpu_topology.os.sched_setaffinity") as pin):
            topo = benchmark_topology()
        pin.assert_called_once_with(0, selected)
        self.assertEqual(hygon_pool(dict(full, physical_core_cpus=[7, 33, 100])), [33, 100])
        previous = set()
        for total in (8, 16, 32):
            width = total // 8
            args = NS(replicas=8, cores_per_replica=width, tp=1, full_frontend_pools=True,
                      concurrency_per_replica=16, text_only=False, renderer_num_workers=None,
                      media_loading_threads=None)
            replicas = make_plan(args, topo, None)["replicas"]
            used = {c for r in replicas for c in r["cpus"]}
            self.assertTrue(previous <= used <= set(selected))
            self.assertEqual(len(used), total)
            self.assertEqual(len(used.intersection(nodes[2])), total//2)
            for r in replicas:
                self.assertEqual(len(r["nodes"]), 1)
                self.assertEqual(r["frontend"], dict(api_server_count=1, renderer_num_workers=width,
                                                      media_loading_threads=width))
            previous = used

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
