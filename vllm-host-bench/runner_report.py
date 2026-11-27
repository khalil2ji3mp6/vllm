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
    single = all(p.get("dp_size") == p.get("tp_size") == 1 for p in profiles.values())
    # A sweep can contain a single-core baseline and multiple DP8 columns.
    worker_scope = "跨 DP 并行，步内同步"
    if any(p.get("tp_size", 1) > 1 for p in profiles.values()):
        worker_scope = "跨 worker 进程并行，步内同步"
    for phase in PHASES:
        rows = ROWS if phase == "prefill" else ROWS + SPEC_ROWS
        header = ["图示对应", "步骤"]
        if not single:
            header.append("多核配置的并行位置")
        columns = []
        for cores, phases in profiles.items():
            header.extend([f"{cores} 核耗时（µs/次）", f"{cores} 核占比"])
            group = phases[phase]
            if phase == "decode":
                # Preserve the original fixtures in JSON. Only the displayed
                # sum combines ordinary decode and isolated speculative calls.
                group = group | phases.get("speculative", {})
            means = {key: group[key]["wall_sum_ns"] / group[key]["calls"] / 1000
                     for _, _, key in rows if key in group and group[key]["calls"]}
            columns.append((means, sum(means.values())))
        lines = [f"## {phase.capitalize()}", "", "| " + " | ".join(header) + " |",
                 "|---|" + "---|" * (len(header) - 1)]
        for parent, step, key in rows:
            measured = any(key in means for means, _ in columns)
            cells = [parent, step]
            if not single:
                cells.append(worker_scope if measured else "—")
            for means, total in columns:
                if key in means:
                    cells.extend([f"{means[key]:,.2f}", f"{100 * means[key] / total:.2f}%" if total else "—"])
                else:
                    cells.extend(["—", "—"])
            lines.append("| " + " | ".join(cells) + " |")
        cells = ["**合计**", "worker 实测项均值之和"]
        if not single:
            cells.append("—")
        for means, total in columns:
            cells.extend([f"**{total:,.2f}**" if means else "—", "100%" if total else "—"])
        lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    report = "\n\n".join(sections) + "\n"
    (out / "runner-profile.json").write_text(json.dumps(profiles, indent=2) + "\n")
    (out / "runner-profile.md").write_text(report)
    return report
