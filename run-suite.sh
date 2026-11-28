#!/usr/bin/env bash
# Run all three host benchmarks with the same core counts on both architectures.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
bench="$root/vllm-host-bench"
python="$bench/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "请先准备已有的无卡测试环境：$python" >&2
    exit 2
fi

# cpu_topology() respects the caller's cpuset and selects one SMT sibling per core.
topology=$(PYTHONPATH="$bench" "$python" - <<'PY'
import platform
from pathlib import Path

from cpu_topology import benchmark_topology

machine = platform.machine().lower()
topology = benchmark_topology()
physical = set(topology["physical_core_cpus"])
full = len(physical)
if full < 32:
    raise SystemExit(f"测试需要 32 个可用物理核，当前核池可用 {full} 个")

cpuinfo = Path("/proc/cpuinfo").read_text()
model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
              if line.partition(":")[0].strip() == "model name"), "")
local = []
if machine == "x86_64":
    if "hygon" not in cpuinfo.lower():
        raise SystemExit("当前 x86 主机未识别为海光，无法套用海光 NUMA 2、6 测试")
    label = "海光 x86"
    # benchmark_topology limited each of the two nodes to at most 16 cores.
    local = sorted(physical)
elif machine in ("aarch64", "arm64"):
    label = "ARM"
else:
    raise SystemExit(f"不支持的架构 {machine}")

print(label)
print(full)
print(model or machine)
print(",".join(map(str, local)))
PY
)
mapfile -t fields <<< "$topology"
platform=${fields[0]}
full=${fields[1]}
model=${fields[2]}
local32=${fields[3]:-}

results="$root/host-bench-results"
mkdir -p "$results"
run=$(mktemp -d "$results/$(date +%Y%m%d-%H%M%S)-XXXXXX")
report="$run/summary.md"
{
    printf '# vLLM 无卡 CPU 基准 · %s\n\n' "$platform"
    printf '%s · %s · %s · %s 个可用物理核\n\n' "$(hostname)" "$model" "$(date '+%F %T %z')" "$full"
    printf '原版单算子用 1、8、16、32 核。整网用 8、16、32 核，DP8/TP1，每 DP 并发 16，32K 输入、1K 输出。新版单算子在同一物理核依次测 32K、128K、200K 输入。\n'
    printf '整网每 DP 一个 API，三个档位每 DP 分配 1、2、4 核，renderer 与媒体池各配 1、2、4 个线程，按阶段运行。\n'
    if [[ -n "$local32" ]]; then
        printf '\n海光 NUMA 2、6 各取 16 个物理核，taskset CPU 列表为 `%s`，内存沿用系统分配策略。\n' "$local32"
    fi
} > "$report"

run_case() {
    local id=$1 heading=$2 md=$3
    shift 3
    local dir="$run/$id" log="$run/$id.log"
    printf '运行 %s（%s）\n' "$id" "$heading"
    if [[ -n "$local32" ]]; then set -- taskset -c "$local32" "$@"; fi
    if ! "$@" --out "$dir" > "$log" 2>&1; then
        tail -n 25 "$log" >&2
        printf '测试失败，请查看 %s\n' "$log" >&2
        return 1
    fi
    if [[ ! -s "$dir/$md" ]]; then
        printf '缺少结果文件 %s/%s，请查看 %s\n' "$dir" "$md" "$log" >&2
        return 1
    fi
    printf '\n### %s\n\n' "$heading" >> "$report"
    sed -E 's/^(#+) /##\1 /' "$dir/$md" >> "$report"
    printf '\n' >> "$report"
}

printf '本次结果目录：%s\n' "$run"
printf '\n## 1. 原版单算子\n' >> "$report"
run_case 01-stages "1、8、16、32 核" stages.md ./run-stages.sh --core-sweep 1 8 16 32

printf '\n## 2. 原版整网\n' >> "$report"
run_case 02-host "DP8 · 8、16、32 核" coverage.md ./run.sh --dp 8 --core-sweep 8 16 32 \
    --concurrency-per-replica 16 --max-num-seqs 16

printf '\n## 3. 新版 worker 单算子\n' >> "$report"
printf '\nPrefill 合计包含一次完整输入分词和 worker 的首个 16K 分块。Decode 合计包含普通步和投机样本。\n' >> "$report"
run_case 03-runner "1 核 · 32K、128K、200K" runner-profile.md ./run-stages.sh --runner-profile --prompt-tokens 32k 128k 200k

# Publish only complete runs; concurrent invocations keep separate result directories.
ln -s "$(basename "$run")/summary.md" "$run/latest-link"
mv -Tf "$run/latest-link" "$results/latest.md"
printf '全部完成：%s/latest.md\n' "$results"
