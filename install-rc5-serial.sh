#!/usr/bin/env bash
# Apply the two matching wheels to an installed test kit without network access.
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench="$kit/vllm-host-bench"
wheels="$kit/host-rc5-wheels"
uvbin="$bench/.tools/uv"
if [[ ! -x "$uvbin" ]]; then uvbin=$(command -v uv); fi
"$uvbin" --no-config pip install --offline --no-index --no-deps --python "$bench/.venv/bin/python" \
    "$wheels/tokenizers-1.0.0rc5+host.serial1-cp311-abi3-manylinux_2_28_$(uname -m).whl" \
    "$wheels/transformers-5.14.1+host.rc5.serial1-py3-none-any.whl"
(cd "$bench" && .venv/bin/python -c 'from pathlib import Path; from tokenizer_rc5 import GlmRc5Tokenizer; assert GlmRc5Tokenizer(Path("model")).encode(" hello", False) == [23745]')
