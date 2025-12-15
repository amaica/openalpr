#!/usr/bin/env bash
set -euo pipefail

ALPR_BIN="${ALPR_BIN:-./build/alpr}"
CONFIG="${CONFIG:-./config/openalpr.conf.defaults}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <car_dir> <moto_dir>" >&2
  exit 1
fi

run_dir() {
  local dir="$1"
  local label="$2"
  echo "## ${label}"
  for img in "$dir"/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
    [ -e "$img" ] || continue
    echo "- file: $img"
    "$ALPR_BIN" -c br --config "$CONFIG" --debug -j "$img" | head -n 5
  done
}

run_dir "$1" "car"
run_dir "$2" "moto"

