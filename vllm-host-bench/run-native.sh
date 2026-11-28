#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$root/source-version.env"
if [[ ! -x "$root/.venv/bin/python" || ! -f "$root/.host-src/.bench-commit" ]] \
  || [[ $(cat "$root/.host-src/.bench-commit") != "$commit" ]]; then
  echo "Set up the environment with $root/install-native.sh first." >&2
  exit 2
fi
export PYTHONPATH="$root/.host-src"
export PYTHONHASHSEED=0
export VLLM_HOST_SOURCE_COMMIT="$commit"
if [[ ${1:-} == --single-process ]]; then
    shift
    exec "$root/.venv/bin/python" "$root/single_process.py" "$@"
fi
exec "$root/.venv/bin/python" "$root/bench.py" --model-metadata "$root/model" "$@"
