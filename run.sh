#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python="$kit/vllm-host-bench/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Run $kit/install.sh first." >&2
  exit 2
fi
exec "$python" "$kit/vllm-host-bench/host_capacity.py" "$@"
