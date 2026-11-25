"""Summarize CPU charged to real service roles from retained benchmark samples."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from report import IMAGE_TOTAL, merge_intervals

# Enclosing timers only. A wait spanning an await can charge unrelated coroutine work.
# Consider transport waits separately in the latency tables.
GROUPS = [
    (
        "前端校验、建状态",
        "frontend",
        ("request_validate", "input_validate_and_enqueue"),
        "请求接入、参数校验",
    ),
    (
        "模板与分词",
        "frontend",
        ("chat_template", "tokenize"),
        "Chat Template、文本 Tokenizer",
    ),
    ("图片接入", "frontend", ("image_base64_decode", "image_decode_rgb"), IMAGE_TOTAL),
    (
        "引擎收包与入队",
        "engine",
        ("engine_request_deserialize", "engine_add_request_and_hash"),
        None,
    ),
    ("调度与 KV 分配", "engine", ("schedule",), "请求调度、batch 组织、token 预算分配"),
    (
        "worker 输入准备",
        "worker",
        ("worker_prepare_cpu",),
        "准备输入 tensor、位置、序列长度和块表",
    ),
    ("引擎更新与输出", "engine", ("engine_output_postprocess",), None),
    ("前端输出处理", "frontend", ("front_output_postprocess",), None),
]


def write_profile(summaries, out):
    trials = [s["workload"]["cold"] for s in summaries]
    wall = sum(
        max(r["end_ns"] for r in group) - min(r["start_ns"] for r in group)
        for group in zip(*trials, strict=True)
    )
    requests = sum(len(r["host_request_latency_ns"]) for rows in trials for r in rows)
    if not wall or not requests:
        raise ValueError("Expected completed service requests")
    result = []
    lines = [
        "| 阶段 | CPU 流程组 | CPU µs/完成请求 | 整轮平均占核 | 该组活跃时平均占核 |",
        "|---|---|---:|---:|---:|",
    ]
    for section, phase in (("Prefill", "prefill_inclusive"), ("Decode", "decode")):
        for label, role, stages, source in GROUPS:
            matched = [
                r
                for s in summaries
                for r in s["timings"]
                if r["case"] == "cold"
                and r["phase"] == phase
                and r["stage"] in stages
                and (
                    r["process"].startswith("worker")
                    if role == "worker"
                    else r["process"] == role
                )
            ]
            if not matched:
                continue
            cpu = sum(r["thread_cpu_sum_ns"] for r in matched)
            active = 0
            if source and section == "Prefill":
                for rounds in zip(
                    *(s["stage_intervals"]["cold"] for s in summaries), strict=True
                ):
                    spans = merge_intervals(
                        span
                        for batch in rounds
                        for span in batch[section].get(source, ())
                    )
                    active += sum(end - start for start, end in spans)
            result.append(
                dict(
                    section=section,
                    group=label,
                    cpu_us_per_request=cpu / requests / 1000,
                    whole_round_cores=cpu / wall,
                    active_cores=cpu / active if active else None,
                )
            )
            during = f"{cpu / active:.2f}" if active else "—"
            lines.append(
                f"| {section} | {label} | {cpu / requests / 1000:.2f} | {cpu / wall:.2f} | {during} |"
            )
    resources = defaultdict(Counter)
    for trials_per_dp in trials:
        for r in trials_per_dp:
            for stage, counts in r.get("prefill_resources", {}).items():
                resources[stage].update(counts)
    if resources:
        lines.extend(
            [
                "",
                "| Prefill 单次调用 | 调用数 | 用户态 µs | 内核态 µs | 次缺页/调用 | 主缺页/调用 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for stage, counts in sorted(resources.items()):
            n = counts["calls"]
            lines.append(
                f"| {stage} | {n} | {counts['user_us'] / n:.2f} | {counts['system_us'] / n:.2f} | "
                f"{counts['minor_faults'] / n:.2f} | {counts['major_faults'] / n:.2f} |"
            )
        (out / "prefill-resources.json").write_text(
            json.dumps(resources, indent=2) + "\n"
        )
    (out / "core-usage.md").write_text("\n".join(lines) + "\n")
    (out / "core-usage.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Retained .runs/cores-N directory")
    args = parser.parse_args()
    samples = [
        json.loads(p.read_text())
        for p in sorted(args.run.glob("replica-*/summary.json"))
    ]
    expected = len(json.loads((args.run / "cpu-plan.json").read_text())["replicas"])
    if len(samples) != expected:
        parser.error(f"Found {len(samples)} of {expected} replicas")
    write_profile(samples, args.run)
