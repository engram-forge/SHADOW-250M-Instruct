#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
output=${1:-"$root/build/linux-arm64/shadow"}
build_dir=$(dirname "$output")
compiler=${CXX:-}
if [[ -z "$compiler" ]]; then
  if command -v clang++-18 >/dev/null; then
    compiler=clang++-18
  else
    compiler=c++
  fi
fi

case "$(uname -m)" in
  aarch64|arm64) ;;
  *) echo "build_linux_arm64.sh requires a native ARM64 Linux environment" >&2; exit 2 ;;
esac

cmake -S "$root/native" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DSHADOW_H618=ON -DCMAKE_CXX_COMPILER="$compiler"
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure
if [[ "$output" != "$build_dir/shadow" ]]; then
  cp "$build_dir/shadow" "$output"
fi

file "$output"
echo "compiler: $compiler"
"$output" --capabilities

if command -v readelf >/dev/null; then
  newest=$(readelf --version-info "$output" | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1 || true)
  echo "newest glibc symbol: ${newest:-none}"
fi
