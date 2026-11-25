#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
arch=$(uname -m)
case "$(uname -s)/$arch" in
  Linux/x86_64|Linux/aarch64) ;;
  *) echo "This kit requires Linux x86_64 or aarch64." >&2; exit 2 ;;
esac
exec "$kit/vllm-host-bench/install-native.sh" --offline "$kit/vllm-host-offline-$arch"
