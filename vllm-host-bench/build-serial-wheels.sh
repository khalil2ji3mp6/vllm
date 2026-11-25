#!/usr/bin/env bash
# Build the offline overlay on an Internet-connected build host with Rust, maturin and Zig.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${1:?Specify an output wheel directory}"
out=$(cd -- "$1" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
curl -fsSL --retry 3 https://files.pythonhosted.org/packages/56/93/ac5b5c177eaeda69d2d31e7acf8006d4022ad7da4729958ca1851120c3da/tokenizers-1.0.0rc5.tar.gz -o rc5.tar.gz
python3 -c 'import hashlib; assert hashlib.sha256(open("rc5.tar.gz", "rb").read()).hexdigest() == "fb6e30d77b8b2a8ed9ce02c000e81e5841f2aa9d402c3b6019442ec70951e4cc"'
tar xzf rc5.tar.gz
cd tokenizers-1.0.0rc5
patch -p1 < "$here/rc5-serial.patch"
for target in aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu; do
    maturin build --release --locked --zig --target "$target" --compatibility manylinux_2_28 --out "$out"
done
wheel="$tmp/transformers-5.14.1-py3-none-any.whl"
python3 "$here/fetch-transformers-wheel.py" "$wheel"
python3 "$here/build-transformers-rc5-wheel.py" "$wheel" "$out" --serial
