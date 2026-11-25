#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench="$kit/vllm-host-bench"
source "$bench/source-version.env"
if [[ ! -x "$bench/.venv/bin/python" || ! -f "$bench/.host-src/.bench-commit" ]] \
   || [[ $(cat "$bench/.host-src/.bench-commit") != "$commit" ]]; then
    echo "Run $kit/install.sh to prepare the pinned CPU environment." >&2
    exit 2
fi
export PYTHONPATH="$bench/.host-src"
export VLLM_HOST_SOURCE_COMMIT="$commit"
export PYTHONHASHSEED=0
exec "$bench/.venv/bin/python" "$bench/stage_bench.py" "$@"
