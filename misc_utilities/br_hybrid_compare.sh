#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-test_data/br}"
ALPR_BIN="${ALPR_BIN:-./build/alpr}"
ALPR_CONFIG="${ALPR_CONFIG:-./config/openalpr.conf.defaults}"
ALPR_RUNTIME="${ALPR_RUNTIME:-./runtime_data}"

if [ ! -x "$ALPR_BIN" ]; then
  echo "alpr binary not found at $ALPR_BIN" >&2
  exit 1
fi

if [ ! -d "$DIR" ]; then
  echo "directory not found: $DIR" >&2
  exit 1
fi

total=0
br_hits=0
eu_hits=0

for img in "$DIR"/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
  [ -e "$img" ] || continue
  total=$((total+1))
if "$ALPR_BIN" -c br --config "$ALPR_CONFIG" --runtime-dir "$ALPR_RUNTIME" -j "$img" | grep -q '"plate":'; then
    br_hits=$((br_hits+1))
  fi
if "$ALPR_BIN" -c eu -p ad --config "$ALPR_CONFIG" --runtime-dir "$ALPR_RUNTIME" -j "$img" | grep -q '"plate":'; then
    eu_hits=$((eu_hits+1))
  fi
done

echo "files: $total"
echo "br (hybrid) hits: $br_hits"
echo "eu/ad hits: $eu_hits"

