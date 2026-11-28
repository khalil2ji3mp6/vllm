"""Reproduce the 200K core scaling and 64-core input length comparisons."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
from tempfile import mkdtemp

from flow_groups import groups
from report import IMAGE_TOTAL, amortized_stage_values, batch_window

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "cores": [(32, 200), (64, 200)],
    "lengths": [(64, k) for k in (32, 128, 200, 400)],
}
TITLES = {"cores": "200K 输入，32 / 64 物理核", "lengths": "64 物理核，输入长度对比"}


def run_args(mode, cores, length_k):
    runs = 5 if mode == "cores" else 3
    return [str(ROOT / "run.sh"), "--dp", "8", "--tp", "1",
            "--cores-per-replica", str(cores // 8), "--api-server-count", "8",
            "--concurrency-per-replica", "24", "--max-num-seqs", "24",
            "--prompt-tokens", str(length_k * 1024), "--new-tokens", "1024",
            "--chunk-tokens", "16384", "--separate-image-text",
            "--warmup", "1", "--runs", str(runs), "--keep-diagnostics"]


def collect(case_dir, cores):
    raw = case_dir / ".runs" / f"cores-{cores}"
    host = json.loads((raw / "host-summary.json").read_text())
    replicas = [json.loads(p.read_text()) for p in sorted(raw.glob("replica-*/summary.json"))]
    if len(replicas) != 8 or host["cpu_plan"]["assigned_physical_cores"] != cores:
        raise ValueError(f"整网结果核数或 DP 数不完整：{raw}")
    rounds = host["per_round"]
    counts = [r["requests"] for r in rounds]
    count = sum(counts)
    work = [r for s in replicas for r in s["workload"]["cold"]]
    values = amortized_stage_values(replicas, "cold", counts, sum(r["output_tokens"] for r in work))
    metrics = {"Tokenizer 整批摊销（µs/请求）": float(values["Prefill", "tokenize"])}
    windows = {k: [] for k in ("tokenize", IMAGE_TOTAL)}
    for i in range(len(rounds)):
        for key in windows:
            spans = [span for s in replicas for span in s["stage_intervals"]["cold"][i]["Prefill"][key]]
            [(start, end)] = batch_window(spans)
            windows[key].append(end - start)
    metrics["图片解码整批摊销（µs/请求，含批内间隙）"] = sum(windows[IMAGE_TOTAL]) / (1000 * count)
    metrics["端到端吞吐（请求/s，各轮中位数）"] = median(r["requests_per_s"] for r in rounds)
    metrics["整批完成时间（秒，各轮均值）"] = mean(r["wall_s"] for r in rounds)
    for label, field in [("平均首 token 时延（ms）", "host_ttft_ns"),
                         ("平均完整请求时延（ms）", "host_request_latency_ns")]:
        metrics[label] = sum(sum(r[field]) for r in work) / (1e6 * count)
    for label, key, stages in [
        ("分词", "tokenize", {"tokenize"}),
        ("图片解码", IMAGE_TOTAL, {"image_base64_decode", "image_decode_rgb"}),
    ]:
        cpu_ns = sum(t["thread_cpu_sum_ns"] for s in replicas for t in s["timings"]
                     if t["case"] == "cold" and t["process"] == "frontend"
                     and t["phase"] == "prefill" and t["stage"] in stages)
        metrics[f"{label}窗口平均有效核数"] = cpu_ns / sum(windows[key])
    return values, metrics


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1),
            *("| " + " | ".join(row) + " |" for row in rows)]


def comparison(mode, results):
    headers = ["指标", *results]
    lines = [f"## {TITLES[mode]}", ""]
    layout = ("每 DP 一个 API，分词池和图片池各由 4 增至 8 线程，计时 5 轮。"
              if mode == "cores" else "每 DP 一个 API，各池每 API 8 线程，计时 3 轮。")
    lines.extend([layout, "", "### 前端与端到端", ""])
    metrics = [m for _, m in results.values()]
    lines.extend(table(headers, [[key, *(f"{m[key]:,.2f}" for m in metrics)] for key in metrics[0]]))
    for phase, unit in [("Prefill", "µs/请求"), ("Decode", "µs/输出 token")]:
        columns = [{g: float(v[phase, g]) for g in groups(phase)} for v, _ in results.values()]
        totals = [sum(c.values()) for c in columns]
        rows = [[g, *(f"{c[g]:,.2f}（{c[g] / total:.2%}）" for c, total in zip(columns, totals))]
                for g in groups(phase)]
        rows.append(["阶段合计", *(f"{total:,.2f}（100%）" for total in totals)])
        lines.extend(["", f"### {phase}（{unit}，整批摊销）", "", *table(headers, rows)])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cores", "lengths", "all"], default="all", nargs="?")
    mode = parser.parse_args().mode
    modes = list(CASES) if mode == "all" else [mode]
    # Check the largest allocation before starting a 32-core trial on a smaller CPU set.
    subprocess.run([*run_args(modes[0], 64, 200), "--plan-only"], stdout=subprocess.DEVNULL, check=True)
    base = ROOT / "host-bench-results" / f"compare-{mode}"
    base.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="run-", dir=base))
    print(f"本次结果目录 {run}", flush=True)
    lines = ["# vLLM 无卡整网对比", "",
             "DP8/TP1，每 DP 并发 24，整机 192。每请求一张 1024×1024 JPEG，输出 1,024 token，16K 分块。"
             "图文分阶段，预热 1 轮。分词明细已计入文本行，图片解码明细与图片接入行分别展示。", ""]
    for name in modes:
        results = {}
        for cores, k in CASES[name]:
            case = run / f"{name}-{cores}c-{k}k"
            log = case.with_suffix(".log")
            print(f"运行 {case.name}，日志 {log}", flush=True)
            with log.open("w") as stream:
                subprocess.run([*run_args(name, cores, k), "--out", str(case)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            label = f"{cores} 核" if name == "cores" else f"{k}K"
            results[label] = collect(case, cores)
        lines.extend([*comparison(name, results), ""])
    report = run / "summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    link = run / "latest-link"
    link.symlink_to(Path(run.name) / report.name)
    os.replace(link, base / "latest.md")
    print(f"汇总报告 {base / 'latest.md'}")


if __name__ == "__main__":
    main()
