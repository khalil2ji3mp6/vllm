#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python="$kit/vllm-host-bench/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Use the existing host kit environment (./install.sh)." >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$python" "$kit/vllm-host-bench/memory_diag.py" "$@"
