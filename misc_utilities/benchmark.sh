#!/usr/bin/env bash
set -euo pipefail

ALPR_BIN="${ALPR_BIN:-./build/alpr}"
CONFIG="${CONFIG:-./config/openalpr.conf.defaults}"
COUNTRY="${COUNTRY:-br}"

DIR="${1:-}"
if [ -z "$DIR" ]; then
  echo "Usage: ALPR_BIN=./build/alpr CONFIG=./config/openalpr.conf.defaults COUNTRY=br $0 <images_dir>" >&2
  exit 1
fi

if [ ! -x "$ALPR_BIN" ]; then
  echo "alpr binary not found or not executable: $ALPR_BIN" >&2
  exit 1
fi

if [ ! -d "$DIR" ]; then
  echo "directory not found: $DIR" >&2
  exit 1
fi

start_ms=$(date +%s%3N)
total=0
hits=0

for img in "$DIR"/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
  [ -e "$img" ] || continue
  total=$((total+1))
  out=$("$ALPR_BIN" -c "$COUNTRY" --config "$CONFIG" -j "$img" || true)
  if echo "$out" | grep -q '"plate":'; then
    hits=$((hits+1))
  fi
done

end_ms=$(date +%s%3N)
elapsed_ms=$((end_ms - start_ms))

if [ $total -gt 0 ]; then
  avg_ms=$((elapsed_ms / total))
else
  avg_ms=0
fi

fps="0"
if [ $elapsed_ms -gt 0 ]; then
  fps=$(awk -v t="$total" -v ms="$elapsed_ms" 'BEGIN{printf("%.2f", (t*1000.0)/ms)}')
fi

rate="0"
if [ $total -gt 0 ]; then
  rate=$(awk -v h="$hits" -v t="$total" 'BEGIN{printf("%.2f", (h*100.0)/t)}')
fi

echo "dir: $DIR"
echo "total_images: $total"
echo "hits: $hits"
echo "hit_rate_percent: $rate"
echo "total_time_ms: $elapsed_ms"
echo "avg_time_ms: $avg_ms"
echo "fps: $fps"

