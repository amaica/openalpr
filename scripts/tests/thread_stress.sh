#!/usr/bin/env bash
set -euo pipefail

FRAME="${FRAME:-artifacts/tests/frames/frame1.jpg}"
LEVELS="${LEVELS:-2 4 8}"
LOOPS="${LOOPS:-50}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
OUT_DIR="artifacts/logs/thread_stress"
REPORT="artifacts/reports/thread_stress.txt"

mkdir -p "$OUT_DIR" "$(dirname "$REPORT")"

if [[ ! -f "$FRAME" ]]; then
  echo "Frame not found: $FRAME" >&2
  exit 1
fi

BIN="./build/src/alpr"
CONF="${CONF:-artifacts/config_video_test/openalpr.conf}"
COUNTRY="${COUNTRY:-br}"
if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN" >&2
  exit 1
fi

echo "thread_stress start" > "$REPORT"
echo "frame=$FRAME" >> "$REPORT"
echo "loops=$LOOPS" >> "$REPORT"
echo "levels=$LEVELS" >> "$REPORT"
echo "timeout_sec=$TIMEOUT_SEC" >> "$REPORT"
echo "config=$CONF country=$COUNTRY" >> "$REPORT"
echo "" >> "$REPORT"

run_level() {
  local level="$1"
  local logdir="$OUT_DIR/N${level}"
  mkdir -p "$logdir"
  local start_ts
  start_ts=$(date +%s)
  local pids=()
  local idx=1
  while [[ "$idx" -le "$level" ]]; do
    (
      for _ in $(seq 1 "$LOOPS"); do
        "$BIN" -j -c "$COUNTRY" --config "$CONF" "$FRAME" >/dev/null
      done
    ) >"$logdir/worker_${idx}.log" 2>&1 &
    pids+=($!)
    idx=$((idx+1))
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=$((failed+1))
    fi
  done

  local end_ts
  end_ts=$(date +%s)
  local wall=$((end_ts - start_ts))
  local passed=$((level - failed))
  {
    echo "LEVEL N=$level wall_time_sec=$wall passed_workers=$passed failed_workers=$failed"
    if [[ "$failed" -gt 0 ]]; then
      local first_fail
      first_fail=$(ls -1 "$logdir"/worker_*.log 2>/dev/null | head -n1)
      echo "first_error_log=$first_fail"
    fi
  } >> "$REPORT"
}

for lvl in $LEVELS; do
  run_level "$lvl"
done

echo "thread_stress done" >> "$REPORT"

