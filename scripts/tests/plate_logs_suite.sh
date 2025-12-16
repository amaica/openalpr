#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
VIDEOS=(
  "/home/aurelio/FONTES/python/SynkiSentinel/videos/output2.avi"
  "/home/aurelio/FONTES/python/SynkiSentinel/videos/output3.avi"
  "/home/aurelio/FONTES/python/SynkiSentinel/videos/output4.avi"
)
CONF="${CONF:-${ROOT}/artifacts/config_video_test/openalpr.conf}"

mkdir -p "${ROOT}/artifacts/logs" "${ROOT}/artifacts/reports"

echo "[suite] building release..."
cmake -S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${BUILD_DIR}" -j"$(nproc)" >/dev/null

PLATE_BIN="${BUILD_DIR}/src/alpr-tool"

report_txt="${ROOT}/artifacts/reports/plate_logs_report.txt"
> "${report_txt}"

for vid in "${VIDEOS[@]}"; do
  base="$(basename "${vid}")"
  plate_log="${ROOT}/artifacts/logs/${base}_plates.log"
  console_log="${ROOT}/artifacts/logs/${base}_console.log"
  echo "[suite] running ${base}"
  "${PLATE_BIN}" preview \
    --conf "${CONF}" \
    --source "${vid}" \
    --log-plates=1 \
    --log-plates-every-n=10 \
    --log-plates-file "${plate_log}" \
    > "${console_log}" 2>&1 || true

  [ -f "${plate_log}" ] || touch "${plate_log}"

  plates=$(grep -c "plate=" "${plate_log}" || true)
  none=$(grep -c "plate=<none>" "${plate_log}" || true)
  echo "${base}: total_lines=${plates} none=${none}" >> "${report_txt}"
  grep -o "reason=[a-zA-Z0-9_]*" "${plate_log}" | sort | uniq -c | sort -nr | head -n 10 >> "${report_txt}" || true
  echo "---" >> "${report_txt}"
done

echo "[suite] done. Report: ${report_txt}"

