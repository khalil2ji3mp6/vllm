#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench="$kit/vllm-host-bench"
if [[ ! -x "$bench/.venv/bin/python" || ! -f "$bench/.host-src/.bench-commit" ]]; then
  echo "Run $kit/install.sh first." >&2
  exit 2
fi
export PYTHONPATH="$bench/.host-src"
exec "$bench/.venv/bin/python" "$bench/tokenizer_bench.py" --model "$bench/model" "$@"
