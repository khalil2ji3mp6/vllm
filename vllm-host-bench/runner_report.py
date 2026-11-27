"""Worker-step statistics. Pool occurrences, never add nested or inclusive spans."""
import json
from pathlib import Path

# Whole-host workers measure these six rows. Isolated Decode also measures SPEC_ROWS.
ROWS = (
    ("prepare_inputs", "_update_states", "worker_batch_update"),
    ("prepare_inputs", "prepare_inputs", "worker_cpu_metadata"),
    ("prepare_inputs", "padding process", "runner_padding"),
    ("prepare_inputs", "build attn metadata", "runner_attn"),
    ("prepare_inputs", "preprocess", "runner_preprocess"),
    ("set_ascend_forward_context", "set_ascend_forward_context", "runner_context"),
)
MEASURED = tuple(key for _, _, key in ROWS)
SPEC_ROWS = (
    ("model_forward", "model_forward", "model_forward"),
    ("compute logits", "compute logits", "compute_logits"),
    ("rej sample", "rej sample", "rej_sample"),
    ("propose", "padding process for draft", "draft_padding"),
    ("propose", "propose", "propose"),
)
PHASES = ("prefill", "decode")


def collect_workers(summaries):
    totals = {phase: {} for phase in PHASES}
    expected = {phase: 0 for phase in PHASES}
    for summary in summaries:
        for sample in summary["workload"]["cold"]:
            for phase in PHASES:
                n = sample.get(phase + "_steps", 0)
                if phase == "prefill":
                    n += sample.get("mixed_steps", 0)
                expected[phase] += n * summary["parameters"]["tp"]
        for row in summary["timings"]:
            phase, key = row["phase"], row["stage"]
            if (row["case"] != "cold" or not row["process"].startswith("worker")
                    or phase not in ("prefill", "mixed", "decode") or key not in MEASURED):
                continue
            group = totals["prefill" if phase == "mixed" else phase]
            sums = group.setdefault(key, {"calls": 0, "wall_sum_ns": 0})
            for field in sums:
                sums[field] += row[field]
    for phase, group in totals.items():
        if any(group.get(key, {}).get("calls", 0) != expected[phase] for key in MEASURED):
            raise ValueError(f"Incomplete {phase} worker profile (expected {expected[phase]} steps)")
    return dict(totals, dp_size=len(summaries), tp_size=summaries[0]["parameters"]["tp"])


def write_tables(profiles, out: Path):
    sections = []
    service_layout = all("frontend_tokenizer" in phases for phases in profiles.values())
    # A sweep can contain a single-core baseline and multiple DP8 columns.
    worker_scope = "跨 DP 并行，步内同步"
    if any(p.get("tp_size", 1) > 1 for p in profiles.values()):
        worker_scope = "跨 worker 进程并行，步内同步"
    elif all(p.get("dp_size", p.get("service_plan", {}).get("dp_size")) == 1 for p in profiles.values()):
        worker_scope = "单 DP、单 worker，步内同步"
    for phase in PHASES:
        rows = ROWS if phase == "prefill" else ROWS + SPEC_ROWS
        header = ["图示对应", "步骤", "多核配置的并行位置"]
        columns = []
        for cores, phases in profiles.items():
            header.extend([f"{cores} 核耗时（µs/次）", f"{cores} 核占比"])
            if service_layout:
                header.append(f"{cores} 核吞吐（次/s）")
            group = phases[phase]
            if phase == "decode":
                # Preserve the original fixtures in JSON. Only the displayed
                # sum combines ordinary decode and isolated speculative calls.
                group = group | phases.get("speculative", {})
            means = {key: group[key]["wall_sum_ns"] / group[key]["calls"] / 1000
                     for _, _, key in rows if key in group and group[key]["calls"]}
            columns.append((means, sum(means.values()), group))
        lines = [f"## {phase.capitalize()}", "", "| " + " | ".join(header) + " |",
                 "|---|---|---|" + "---:|" * (len(header) - 3)]
        if phase == "prefill" and service_layout:
            cells = ["前端（独立测量）", "Tokenizer", "跨请求多核，renderer 线程池"]
            for phases in profiles.values():
                tok = phases["frontend_tokenizer"]
                cells.extend([f"{tok['wall_sum_ns'] / tok['calls'] / 1000:,.2f}", "—", f"{tok['ops_s']:,.2f}"])
            lines.append("| " + " | ".join(cells) + " |")
        for parent, step, key in rows:
            measured = any(key in means for means, _, _ in columns)
            cells = [parent, step, worker_scope if measured else "—"]
            for means, total, group in columns:
                if key in means:
                    cells.extend([f"{means[key]:,.2f}", f"{100 * means[key] / total:.2f}%" if total else "—"])
                else:
                    cells.extend(["—", "—"])
                if service_layout:
                    cells.append(f"{group[key]['ops_s']:,.2f}" if key in group else "—")
            lines.append("| " + " | ".join(cells) + " |")
        cells = ["**合计**", "worker 实测项均值之和" if service_layout else "实测项均值之和", "—"]
        for means, total, _ in columns:
            cells.extend([f"**{total:,.2f}**" if means else "—", "100%" if total else "—"])
            if service_layout:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
        if service_layout and phase == "prefill":
            lines.extend(["", "Tokenizer 计每个 32K 请求的分词执行时间，吞吐含线程池排队。原生 encode 可跨请求用多核，导出 Python IDs 时同一 API 内的线程共享 GIL。其余行计每个 worker 步，合计和占比仅含 worker 行。TP1 每 DP 一个 worker，步内同步，DP 间并行。"])
            layouts = []
            for cores, phases in profiles.items():
                dp = phases["service_plan"]["replicas"]
                widths = sorted({(r["frontend"]["api_server_count"], r["frontend"]["renderer_num_workers"])
                                 for r in dp})
                config = "、".join(f"{apis} API/DP × {threads} 线程/API" for apis, threads in widths)
                peaks = phases["frontend_tokenizer"]["peak_by_api"]
                layouts.append(f"{cores} 核 {len(dp)} DP × {config}（实测每 API 重叠 {min(peaks)}–{max(peaks)}）")
            lines.extend(["", "分词布局　" + " / ".join(layouts)])
        sections.append("\n".join(lines))
    report = "\n\n".join(sections) + "\n"
    (out / "runner-profile.json").write_text(json.dumps(profiles, indent=2) + "\n")
    (out / "runner-profile.md").write_text(report)
    return report
