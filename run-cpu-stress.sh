#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench="$kit/vllm-host-bench"
python="$bench/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  python=python3
fi
exec "$python" "$bench/cpu_stress.py" "$@"
