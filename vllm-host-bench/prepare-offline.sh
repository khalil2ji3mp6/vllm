#!/usr/bin/env bash
# Download a Linux toolchain and locked wheels on a machine with Internet access.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
arch=${1:?Usage: prepare-offline.sh x86_64|aarch64 OUTPUT_DIRECTORY}
output=${2:?Supply an output directory}
case "$arch" in x86_64|aarch64) ;; *) echo "Unsupported architecture: $arch" >&2; exit 2;; esac
source "$root/source-version.env"
mkdir -p "$output/wheels"
cd "$output"

fetch() {
  if [[ ! -f "$2" ]]; then
    curl -fLsS --retry 3 "$1" -o "$2.part"
    mv "$2.part" "$2"
  fi
}

fetch "https://github.com/astral-sh/uv/releases/download/0.13.9/uv-$arch-unknown-linux-gnu.tar.gz" uv.tar.gz
fetch "https://releases.astral.sh/github/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-$arch-unknown-linux-gnu-install_only_stripped.tar.gz" python.tar.gz
fetch "https://codeload.github.com/vllm-project/vllm/tar.gz/$commit" source.tar.gz
if command -v sha256sum >/dev/null 2>&1; then
  hash=(sha256sum)
else
  hash=(shasum -a 256)
fi
printf '%s  source.tar.gz\n' "$source_sha256" | "${hash[@]}" -c -

platforms=(--platform "manylinux2014_$arch")
for minor in {28..17}; do
  platforms+=(--platform "manylinux_2_${minor}_$arch")
done
if [[ $arch == x86_64 ]]; then
  platforms+=(--platform manylinux2010_x86_64 --platform manylinux1_x86_64)
fi
wheel_options=(--dest wheels --only-binary=:all: --no-deps
  --python-version 3.12.16 --implementation cp --abi cp312 "${platforms[@]}")
grep -vE '^(torch|transformers)==' "$root/requirements-host.lock" > pypi-requirements.txt
python3 -m pip --isolated --disable-pip-version-check download "${wheel_options[@]}" \
  --index-url https://pypi.org/simple -r pypi-requirements.txt
python3 -m pip --isolated --disable-pip-version-check download "${wheel_options[@]}" \
  --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' "$root/requirements-host.lock")"
upstream=$(mktemp -d)
trap 'rm -rf "$upstream"' EXIT
python3 "$root/fetch-transformers-wheel.py" "$upstream/transformers-5.14.1-py3-none-any.whl"
python3 "$root/build-transformers-rc5-wheel.py" "$upstream/transformers-5.14.1-py3-none-any.whl" wheels
printf '%s\n' "$arch" > architecture.txt
"${hash[@]}" uv.tar.gz python.tar.gz source.tar.gz architecture.txt wheels/*.whl > SHA256SUMS
