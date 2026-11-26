"""Write the two user-facing benchmark tables."""
import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

from flow_groups import TARGETS, TEXT, groups


def point(process, stage):
    return process, stage


IMAGE_TOTAL = "图片接入与编解码合计"
LEGACY_IMAGE_ROWS = ("图片／视频读取与解码", "图片 Base64 解码", "图片接入整批", "图片编码（独立样本）")
IMAGE_ROWS = [
    (IMAGE_TOTAL, "CPU",
     [point("frontend", "image_decode_rgb"), point("frontend", "image_base64_decode")]),
    ("图片 Base64 编码（独立样本）", "CPU",
     [point("frontend", "image_base64_encode")]),
]

PREFILL = [
    ("请求接入、参数校验", "CPU",
     [point("frontend", "request_validate"), point("frontend", "input_validate_and_enqueue")]),
    ("Chat Template、文本 Tokenizer", "CPU",
     [point("frontend", "chat_template"), point("frontend", "tokenize")]),
    *IMAGE_ROWS,
    ("多模态 Processor（resize、归一化、patch 等）", "CPU，部分模型为 CPU＋卡", []),
    ("前端与引擎间 IPC、请求入队", "CPU",
     [point("frontend", "front_engine_encode_send"),
      point("engine", "engine_request_deserialize"),
      point("engine", "engine_add_request_and_hash")]),
    ("请求调度、batch 组织、token 预算分配", "CPU",
     [point("engine", "schedule")]),
    ("前缀缓存匹配、KV 逻辑块分配", "CPU",
     [point("engine", "prefix_lookup"), point("engine", "kv_allocate_or_extend")]),
    ("准备输入 tensor、位置、序列长度和块表", "CPU＋卡",
     [point("worker", "worker_prepare_cpu")]),
    ("输入及元数据 H2D", "CPU → 卡", []),
    ("算子下发、执行图启动", "CPU", []),
    ("视觉 Encoder、特征投影", "卡", []),
    ("Token Embedding、GEMM、Norm、RoPE、FFN 等", "卡", []),
    ("Prefill Attention 计算", "卡", []),
    ("写入新 KV Cache、读取已缓存的 KV", "卡", []),
    ("多卡集合通信（AllReduce、AllGather 等）", "CPU＋卡，跨机时可经网卡", []),
    ("LM Head、logits 计算", "卡", []),
    ("常规 top-k／top-p、随机／贪心采样", "卡", []),
    ("回传 token ID 和所需 logprobs", "卡 → CPU",
     [point("worker0", "worker_result_enqueue")]),
    ("请求状态更新、停止条件检查", "CPU",
     [point("engine", "update_request_and_output"), point("engine", "check_stop")]),
    ("Detokenize、文本停止串检查、响应序列化", "CPU",
     [point("frontend", "detokenize_and_stop_strings"),
      point("frontend", "serialize_stream_chunks")]),
    ("可选：PD 控制消息、传输元数据交互", "CPU", []),
    ("可选：KV 卸载／加载、向 Decode 端传输 Prefill KV", "CPU＋卡，跨机时可经网卡", []),
    ("补充：Core 与 worker 的 Host IPC", "CPU",
     [point("engine", "engine_worker_enqueue"),
      point("engine", "engine_worker_reply_wait")]),
    ("补充：Prefill 前处理合计", "CPU",
     [point("frontend", "front_prefill_batch")]),
    ("补充：Prefill 后处理合计", "CPU",
     [point("engine", "engine_output_postprocess"),
      point("frontend", "front_output_postprocess")]),
]

DECODE = [
    ("本轮调度、batch 更新、进程间控制消息", "CPU",
     [point("engine", "schedule"), point("worker", "worker_batch_update"),
      point("engine", "engine_worker_enqueue"),
      point("engine", "engine_worker_reply_wait")]),
    ("KV 逻辑块扩展、块表管理", "CPU",
     [point("engine", "kv_allocate_or_extend")]),
    ("准备上轮 token、位置、序列长度等输入", "CPU＋卡",
     [point("worker", "worker_cpu_metadata")]),
    ("必要的输入及元数据 H2D", "CPU → 卡", []),
    ("算子下发、执行图启动", "CPU", []),
    ("Token Embedding、GEMM、Norm、RoPE、FFN 等", "卡", []),
    ("读取历史 KV Cache", "卡", []),
    ("Decode Attention 计算", "卡", []),
    ("写入当前 token 的新 KV", "卡", []),
    ("多卡集合通信（AllReduce、AllGather 等）", "CPU＋卡，跨机时可经网卡", []),
    ("LM Head、logits 计算", "卡", []),
    ("常规 top-k／top-p、随机／贪心采样", "卡", []),
    ("回传 token ID 和所需 logprobs", "卡 → CPU",
     [point("worker0", "worker_result_enqueue")]),
    ("请求状态更新、EOS／长度等停止条件检查", "CPU",
     [point("engine", "update_request_and_output"), point("engine", "check_stop")]),
    ("增量 Detokenize、文本停止串检查", "CPU",
     [point("frontend", "detokenize_and_stop_strings")]),
    ("流式响应序列化、网络返回", "CPU",
     [point("frontend", "serialize_stream_chunks")]),
    ("请求结束后释放 KV 逻辑块", "CPU", [point("engine", "kv_free")]),
    ("可选：PD 控制消息、传输元数据交互", "CPU", []),
    ("可选：KV 卸载／加载、接收远端 Prefill KV", "CPU＋卡，跨机时可经网卡", []),
    ("补充：Decode 后处理合计", "CPU",
     [point("engine", "engine_output_postprocess"),
      point("frontend", "front_output_postprocess")]),
]

LABELS = ("流程和特性", "昇腾 NPU 平台执行位置")


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def capture_stage_intervals(processes):
    spans = defaultdict(list)
    for process, phases in processes.items():
        names = (process, "worker") if process.startswith("worker") else (process,)
        for (phase, stage), samples in phases.items():
            # A mixed scheduling step has one shared wall interval; attribute it once.
            if phase in ("prefill", "mixed"):
                section = "Prefill"
            elif phase == "decode":
                section = "Decode"
            else:
                continue
            for name in names:
                spans[section, name, stage].extend((start, end) for start, end, _ in samples)

    captured = {
        section: {
            title: merge_intervals(span for target in targets
                                   for span in spans.get((section, *target), ()))
            for title, _, targets in design if targets
        }
        for section, design in (("Prefill", PREFILL), ("Decode", DECODE))
    }


    for section, rows in captured.items():
        for group in groups(section):
            rows[group] = merge_intervals(span for proc, stage in TARGETS[group]
                                          for span in spans.get((section, proc, stage), ()))
    return captured


def stage_values(summaries, case):
    if any("stage_intervals" not in s for s in summaries):
        raise ValueError("This report needs a new run with per-round wall-clock intervals")
    counts = {len(s["workload"][case]) for s in summaries}
    if len(counts) != 1 or not next(iter(counts), 0):
        raise ValueError(f"Replicas must have the same nonzero round count for {case}")
    rounds = counts.pop()
    if any(len(s["stage_intervals"][case]) != rounds for s in summaries):
        raise ValueError(f"Stage intervals do not match measured rounds for {case}")

    completed = sum(len(r["host_request_latency_ns"]) for s in summaries for r in s["workload"][case])
    if not completed:
        raise ValueError(f"No completed requests for {case}")
    # Each round shares one host monotonic clock. Merge across DPs separately
    # in each round so overlapping work uses one unit of elapsed time.
    tables = {"Prefill": [], "Decode": []}
    for phase, design in (("Prefill", PREFILL), ("Decode", DECODE)):
        for title, location, targets in design:
            titles = (title, *LEGACY_IMAGE_ROWS) if title == IMAGE_TOTAL else (title,)
            seen = False
            duration = 0
            for index in range(rounds):
                intervals = merge_intervals(
                    span for s in summaries
                    for name in titles
                    for span in s["stage_intervals"][case][index][phase].get(name, ())
                )
                seen |= bool(intervals)
                duration += sum(end - start for start, end in intervals)
            value = f"{duration / (completed * 1_000):.2f}" if seen else "—"
            tables[phase].append((title, location, value))
    return tables


def write_detail_tables(columns, out, case="cold"):
    headings, summaries = zip(*columns, strict=True)
    tables = [stage_values(group, case) for group in summaries]
    header = (*LABELS, *headings)
    lines = []
    for phase in ("Prefill", "Decode"):
        rows = []
        for aligned in zip(*(table[phase] for table in tables), strict=True):
            title, location, _ = aligned[0]
            rows.append((title, location, *(row[2] for row in aligned)))
        lines.extend([f"## {phase}", "", "| " + " | ".join(header) + " |",
                      "|" + "---|" * len(header)])
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        lines.append("")
        with (out / f"{phase.lower()}.csv").open("w", newline="", encoding="utf-8-sig") as file:
            csv.writer(file).writerows([header, *rows])
    markdown = "\n".join(lines)
    (out / "coverage.md").write_text(markdown, encoding="utf-8")
    return markdown


def write_stage_tables(columns, out, case="cold", diagnostics=False):
    # Retained pre-group summaries can still be rendered using their original schema.
    if TEXT not in columns[0][1][0]["stage_intervals"][case][0]["Prefill"]:
        return write_detail_tables(columns, out, case)
    if diagnostics:
        detail = out / "details"
        detail.mkdir(exist_ok=True)
        write_detail_tables(columns, detail, case)
    headings = [h for h, _ in columns]
    result = []
    rates = []
    for heading, replicas in columns:
        rounds = list(zip(*(s["workload"][case] for s in replicas), strict=True))
        counts = [sum(len(r["host_request_latency_ns"]) for r in batch) for batch in rounds]
        rates.append(median(n * 1e9 / (max(r["end_ns"] for r in batch) - min(r["start_ns"] for r in batch))
                            for n, batch in zip(counts, rounds, strict=True)))
        values = {}
        for phase in ("Prefill", "Decode"):
            for group in groups(phase):
                samples = []
                seen = False
                for i, n in enumerate(counts):
                    # Overlapping work on different DPs counts once toward host wall time.
                    spans = merge_intervals(span for s in replicas for span in
                                            s["stage_intervals"][case][i][phase][group])
                    seen |= bool(spans)
                    samples.append(sum(b-a for a,b in spans) / (1000*n))
                values[phase, group] = f"{mean(samples):.2f}" if seen else "—"
        result.append(values)
    lines = []
    for phase in ("Prefill", "Decode"):
        rows = [("流程", *headings)]
        rows.extend((g, *(v[phase,g] for v in result)) for g in groups(phase))
        # Sum the displayed precision so the CSV and Markdown can be checked by hand.
        totals = []
        for column in result:
            amounts = [column[phase, g] for g in groups(phase) if column[phase, g] != "—"]
            totals.append(f"{sum(map(Decimal, amounts)):.2f}" if amounts else "—")
        rows.append(("阶段合计", *totals))
        with (out / f"{phase.lower()}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        lines.extend([f"## {phase}", "", "| " + " | ".join(rows[0]) + " |",
                      "|" + "---|" * len(rows[0])])
        lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        lines.append("")
    labels = [h.split("（")[0] + " " if len(columns)>1 else "" for h in headings]
    lines.append("端到端吞吐（请求/s） " + " / ".join(f"{label}{rate:.2f}" for label, rate in zip(labels, rates)))
    markdown = "\n".join(lines) + "\n"
    (out / "coverage.md").write_text(markdown)
    return markdown


def write_coverage(data, out, case=None):
    return write_stage_tables([("摊销墙钟（µs/请求）", [data])], out,
                              case or next(iter(data["workload"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case", choices=("cold", "l1"))
    args = parser.parse_args()
    out = args.out or args.summary.parent
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    if args.case and args.case not in summary["workload"]:
        parser.error(f"Summary has no measurements for {args.case}")
    write_coverage(summary, out, args.case)


if __name__ == "__main__":
    main()
