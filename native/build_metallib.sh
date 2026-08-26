#!/bin/sh
set -eu
source_file=$1
output_file=$2
air_file="${output_file}.air"
if xcrun -sdk macosx metal -c "$source_file" -o "$air_file" 2>/dev/null; then
  xcrun -sdk macosx metallib "$air_file" -o "$output_file"
  rm -f "$air_file"
  echo "built $output_file"
else
  rm -f "$air_file"
  echo "Metal offline toolchain unavailable; runner will compile the shader at first use" >&2
fi
