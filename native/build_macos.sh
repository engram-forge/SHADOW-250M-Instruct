#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=${1:-"$root/../build/macos/shadow"}
mkdir -p "$(dirname -- "$out")"
xcrun clang++ -std=c++20 -O3 -DNDEBUG -arch arm64 -mmacosx-version-min=14.0 \
  -Wall -Wextra -Wpedantic -DSHADOW_APPLE=1 \
  -I"$root/include" \
  "$root/src/main.cpp" "$root/src/model.cpp" "$root/src/archive.cpp" "$root/src/metal_archive.mm" \
  -framework Accelerate -framework Foundation -framework Metal \
  -o "$out"
"$root/build_metallib.sh" "$root/shaders/archive.metal" "${out}.metallib"
echo "built $out"
