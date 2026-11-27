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

from cpu_topology import cpu_topology

machine = platform.machine().lower()
topology = cpu_topology()
physical = set(topology["physical_core_cpus"])
full = len(physical)
if full < 64 or full % 8:
    raise SystemExit(f"可用物理核数为 {full}；64 核档和 DP8 全核档需要至少 64 核且总数为 8 的倍数")

cpuinfo = Path("/proc/cpuinfo").read_text()
model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
              if line.partition(":")[0].strip() == "model name"), "")
local = []
if machine == "x86_64":
    if "hygon" not in cpuinfo.lower():
        raise SystemExit("当前 x86 主机未识别为海光，无法套用海光 NUMA 2、6 测试")
    label = "海光 x86"
    for node in (2, 6):
        cores = sorted(physical.intersection(topology["numa_nodes"].get(node, [])))
        if len(cores) < 16:
            raise SystemExit(f"NUMA {node} 只有 {len(cores)} 个可用物理核，需要 16 个")
        local.extend(cores[:16])
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
multi=(32 64)
if (( full > 64 )); then multi+=("$full"); fi

results="$root/host-bench-results"
mkdir -p "$results"
run=$(mktemp -d "$results/$(date +%Y%m%d-%H%M%S)-XXXXXX")
report="$run/summary.md"
{
    printf '# vLLM 无卡 CPU 基准 · %s\n\n' "$platform"
    printf '%s · %s · %s · %s 个可用物理核\n\n' "$(hostname)" "$model" "$(date '+%F %T %z')" "$full"
    printf '自动选核档位（物理核）：1 %s。整网 1 核用 DP1，其余用 DP8，均为 TP1，每 DP 并发 48，输入 32K，输出 1K。\n' "${multi[*]}"
    if [[ -n "$local32" ]]; then
        printf '\n海光 NUMA 2、6 各取 16 个物理核，taskset CPU 列表为 `%s`，内存沿用系统分配策略。\n' "$local32"
    fi
} > "$report"

run_case() {
    local id=$1 heading=$2 md=$3
    shift 3
    local dir="$run/$id" log="$run/$id.log"
    printf '运行 %s（%s）\n' "$id" "$heading"
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
run_case 01-stages-auto "自动选核" stages.md ./run-stages.sh --core-sweep 1 "${multi[@]}"
if [[ -n "$local32" ]]; then
    run_case 01-stages-local32 "NUMA 2、6 · 32 核" stages.md \
        taskset -c "$local32" ./run-stages.sh --core-sweep 32
fi

printf '\n## 2. 原版整网\n' >> "$report"
run_case 02-host-dp1 "自动选核 · DP1 · 1 核" coverage.md ./run.sh --dp 1 --core-sweep 1
run_case 02-host-dp8 "自动选核 · DP8" coverage.md ./run.sh --dp 8 --core-sweep "${multi[@]}"
if [[ -n "$local32" ]]; then
    run_case 02-host-local32 "NUMA 2、6 · DP8 · 32 核" coverage.md \
        taskset -c "$local32" ./run.sh --dp 8 --core-sweep 32
fi

printf '\n## 3. 新版单算子（runner-profile）\n' >> "$report"
printf '\nDecode 合计为普通 Decode 六项与投机验证五项的独立调用均值之和。\n' >> "$report"
run_case 03-runner-auto "自动选核" runner-profile.md \
    ./run-stages.sh --runner-profile --core-sweep 1 "${multi[@]}"
if [[ -n "$local32" ]]; then
    run_case 03-runner-local32 "NUMA 2、6 · 32 核" runner-profile.md \
        taskset -c "$local32" ./run-stages.sh --runner-profile --core-sweep 32
fi

# Publish only complete runs; concurrent invocations keep separate result directories.
ln -s "$(basename "$run")/summary.md" "$run/latest-link"
mv -Tf "$run/latest-link" "$results/latest.md"
printf '全部完成：%s/latest.md\n' "$results"
