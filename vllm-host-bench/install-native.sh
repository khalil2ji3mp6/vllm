#!/usr/bin/env bash
# Install a project-local Linux host benchmark environment using uv.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
arch=$(uname -m)
case "$(uname -s)/$arch" in
  Linux/aarch64|Linux/x86_64) ;;
  *) echo "Use Linux on aarch64 or x86_64." >&2; exit 2 ;;
esac
offline=""
if (( $# > 0 )); then
  if [[ $# != 2 || $1 != --offline ]]; then
    echo "Usage: $0 [--offline DIRECTORY]" >&2
    exit 2
  fi
  offline=$(cd -- "$2" && pwd)
fi
cd "$root"
source "$root/source-version.env"
found=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
if [[ -z "$found" || "$(printf '%s\n' 2.28 "$found" | sort -V | head -1)" != 2.28 ]]; then
  echo "The pinned CPU dependency wheels require glibc 2.28 or newer." >&2
  exit 2
fi

mkdir -p "$root/.tools"
if [[ -n "$offline" ]]; then
  if [[ $(cat "$offline/architecture.txt") != "$arch" ]]; then
    echo "Offline bundle architecture does not match $arch." >&2
    exit 2
  fi
  (cd "$offline" && sha256sum --quiet -c SHA256SUMS)
  export UV_OFFLINE=1
  uv_bin="$root/.tools/uv"
  tar -xzf "$offline/uv.tar.gz" -C "$root/.tools" --strip-components=1 \
    "uv-$arch-unknown-linux-gnu/uv"
elif command -v uv >/dev/null 2>&1; then
  uv_bin=$(command -v uv)
else
  uv_bin="$root/.tools/uv"
  if [[ ! -x "$uv_bin" ]]; then
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT
    curl -fLsS --retry 2 https://astral.sh/uv/install.sh -o "$installer"
    UV_UNMANAGED_INSTALL="$root/.tools" sh "$installer"
  fi
fi

if [[ ! -f "$root/.host-src/.bench-commit" ]] || [[ $(cat "$root/.host-src/.bench-commit") != "$commit" ]]; then
  if [[ -e "$root/.host-src" ]]; then
    echo "Move the existing .host-src aside before setting up vLLM source commit $commit." >&2
    exit 2
  fi
  staging=$(mktemp -d "$root/.host-src.XXXXXX")
  trap 'rm -rf "$staging"; if [[ -n "${installer:-}" ]]; then rm -f "$installer"; fi' EXIT
  if [[ -n "$offline" ]]; then
    archive="$offline/source.tar.gz"
  else
    archive="$staging/source.tar.gz"
    curl -fLsS --retry 2 "https://codeload.github.com/vllm-project/vllm/tar.gz/$commit" -o "$archive"
  fi
  printf '%s  %s\n' "$source_sha256" "$archive" | sha256sum -c -
  tar -xzf "$archive" -C "$staging" --strip-components=1 \
    "vllm-$commit/vllm" "vllm-$commit/LICENSE"
  if [[ -z "$offline" ]]; then
    rm -f "$archive"
  fi
  printf '%s\n' "$commit" > "$staging/.bench-commit"
  mv "$staging" "$root/.host-src"
fi

export UV_PYTHON_INSTALL_DIR="$root/.tools/python"
python=3.12.16
indexes=(--torch-backend cpu)
if [[ -n "$offline" ]]; then
  runtime="$root/.tools/python-bundle"
  python="$runtime/bin/python3.12"
  if [[ ! -x "$python" ]]; then
    mkdir -p "$runtime"
    tar -xzf "$offline/python.tar.gz" -C "$runtime" --strip-components=1
  fi
  indexes=(--no-index --find-links "$offline/wheels")
fi
if [[ ! -x "$root/.venv/bin/python" ]]; then
  "$uv_bin" --no-config --no-cache venv --python "$python" "$root/.venv"
fi
"$root/.venv/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 12, 16), sys.version'
if [[ -z "$offline" ]]; then
  local_wheels="$root/.tools/wheels"
  mkdir -p "$local_wheels"
  if [[ ! -f "$local_wheels/transformers-5.14.1+host.rc5-py3-none-any.whl" ]]; then
    upstream_wheel="$local_wheels/transformers-5.14.1-py3-none-any.whl"
    "$root/.venv/bin/python" "$root/fetch-transformers-wheel.py" "$upstream_wheel"
    "$root/.venv/bin/python" "$root/build-transformers-rc5-wheel.py" "$upstream_wheel" "$local_wheels"
  fi
  indexes+=(--find-links "$local_wheels")
fi
"$uv_bin" --no-config --no-cache pip sync --python "$root/.venv/bin/python" "${indexes[@]}" \
  --only-binary :all: "$root/requirements-host.lock"
"$root/.venv/bin/python" "$root/prepare-source.py" "$root/.host-src"
"$root/.venv/bin/python" -c 'import torch; assert torch.version.cuda is None and torch.version.hip is None; print("CPU torch", torch.__version__)'
"$root/.venv/bin/python" -c 'from pathlib import Path; from tokenizer_rc5 import GlmRc5Tokenizer; t = GlmRc5Tokenizer(Path("model")); assert t.encode(" hello", add_special_tokens=False) == [23745]; print("GLM-5.2 tokenizer 1.0.0rc5")'
