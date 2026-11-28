#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python="$root/vllm-host-bench/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "请先运行 $root/install.sh 准备测试环境" >&2
    exit 2
fi
exec "$python" "$root/vllm-host-bench/compare_host_runs.py" "$@"
