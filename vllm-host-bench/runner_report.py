"""Worker-step statistics. Pool occurrences, never add nested or inclusive spans."""
import json
from pathlib import Path

# The first six rows execute host code. The last five have no fixture equivalent.
ROWS = (
    ("prepare_inputs", "_update_states", "worker_batch_update"),
    ("prepare_inputs", "prepare_inputs", "worker_cpu_metadata"),
    ("prepare_inputs", "padding process", "runner_padding"),
    ("prepare_inputs", "build attn metadata", "runner_attn"),
    ("prepare_inputs", "preprocess", "runner_preprocess"),
    ("set_ascend_forward_context", "set_ascend_forward_context", "runner_context"),
    ("model_forward", "model_forward", None),
    ("compute logits", "compute logits", None),
    ("rej sample", "rej sample", None),
    ("propose", "padding process for draft", None),
    ("propose", "propose", None),
)
MEASURED = tuple(key for _, _, key in ROWS if key)
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
    return totals


def write_tables(profiles, out: Path):
    sections = []
    for cores, phases in profiles.items():
        for phase, group in phases.items():
            means = {k: v["wall_sum_ns"] / v["calls"] / 1000 for k, v in group.items() if v["calls"]}
            total = sum(means.values())
            lines = [f"## {phase.capitalize()} · {cores} 核", "",
                     "| 图示对应 | 步骤 | 耗时（µs/worker 步） | 耗时占比（%） |",
                     "|---|---|---:|---:|"]
            for parent, step, key in ROWS:
                cells = f"{means[key]:,.2f} | {100 * means[key] / total:.2f}%" if key in means else "— | —"
                lines.append(f"| {parent} | {step} | {cells} |")
            sections.append("\n".join(lines))
    report = "\n\n".join(sections) + "\n"
    (out / "runner-profile.json").write_text(json.dumps(profiles, indent=2) + "\n")
    (out / "runner-profile.md").write_text(report)
    return report
