"""Regression for the same ten requests executing concurrently across DPs."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from flow_groups import TEXT, IPC, IMAGE
from report import amortized_stage_values, capture_stage_intervals, write_stage_tables


def replica(start, end):
    rows = {
        ("prefill", "tokenize"): [(start, end, 0)] * 5,
        ("prefill", "text_after_media"): [(start, end, 0)] * 5,
        ("decode", "front_output_ipc_decode"): [(start, end, 0)] * 5,
    }
    return {"stage_intervals": {"cold": [capture_stage_intervals({"frontend": rows})]}}


class BatchWallTest(unittest.TestCase):
    def test_ten_overlapping_requests_take_one_second(self):
        replicas = [replica(0, 10**9), replica(0, 10**9)]
        result = amortized_stage_values(replicas, "cold", [10], 30)
        self.assertEqual(result["Prefill", "tokenize"], "100000.00")
        self.assertEqual(result["Prefill", TEXT], "100000.00")
        self.assertEqual(result["Decode", IPC], "50000.00")
        self.assertEqual(result["Prefill", IMAGE], "—")

    def test_disjoint_tokenizer_waves_and_independent_rounds(self):
        replicas = [replica(0, 10**8), replica(9*10**8, 10**9)]
        for summary in replicas:
            intervals = summary["stage_intervals"]["cold"]
            intervals.append(intervals[0])
        result = amortized_stage_values(replicas, "cold", [10, 10], 20)
        self.assertEqual(result["Prefill", "tokenize"], "100000.00")
        self.assertEqual(result["Decode", IPC], "—")

    def test_image_between_validation_and_rendering(self):
        rows = {
            ("prefill", "request_validate"): [(0, 1000, 0)],
            ("prefill", "request_image_elapsed"): [(1000, 8000, 0)],
            ("prefill", "text_after_media"): [(8000, 10000, 0)],
        }
        summary = {"stage_intervals": {"cold": [capture_stage_intervals({"frontend": rows})]}}
        result = amortized_stage_values([summary], "cold", [1], 1)
        self.assertEqual(result["Prefill", TEXT], "3.00")
        self.assertEqual(result["Prefill", IMAGE], "7.00")

    def test_request_mean_weights_each_request_and_excludes_first_decode_token(self):
        data = replica(0, 10**9)
        data["batch_wall_version"] = 1
        rounds = []
        for count, duration in [(2, 10**9), (1, 4 * 10**9)]:
            rounds.append(dict(host_request_latency_ns=[duration] * count,
                               start_ns=0, end_ns=duration, output_tokens=count * 3,
                               request_stage_ns={"Prefill": {TEXT: count * duration,
                                                             IMAGE: count * duration // 2},
                                                 "Decode": {IPC: count * 2 * 1000}}))
        data["workload"] = {"cold": rounds}
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_stage_tables([("test", [data])], out, request_mean=True)
            def values(phase):
                with (out / f"{phase}.csv").open(encoding="utf-8-sig") as file:
                    return {name: value for name, scope, value in csv.reader(file)}
            self.assertEqual(values("prefill")[TEXT], "2000000.00")
            self.assertEqual(values("prefill")[IMAGE], "1000000.00")
            self.assertEqual(values("decode")[IPC], "1.00")
