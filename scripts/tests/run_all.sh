#!/usr/bin/env bash
set -euo pipefail

log() { echo "[run_all] $*"; }
die() { echo "[run_all][error] $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_DIR="${CONFIG_DIR:-}"
if [[ -n "${CONFIG_DIR}" ]]; then
  :
elif [[ -d "/opt/alpr" ]]; then
  CONFIG_DIR="/opt/alpr"
elif [[ -d "${REPO_ROOT}/config" ]]; then
  CONFIG_DIR="${REPO_ROOT}/config"
else
  die "CONFIG_DIR not set and no default config directory found (/opt/alpr or ./config). Export CONFIG_DIR explicitly."
fi

TEST_IMAGE="${TEST_IMAGE:-}"
[[ -n "${TEST_IMAGE}" ]] || die "TEST_IMAGE is required (path to an image for smoke test)"
[[ -f "${TEST_IMAGE}" ]] || die "TEST_IMAGE not found: ${TEST_IMAGE}"

YOLO_PT="${YOLO_PT:-/home/aurelio/FONTES/python/SynkiSentinel/models/plates/yolov8n_plates.pt}"
[[ -f "${YOLO_PT}" ]] || die "YOLO_PT not found: ${YOLO_PT}"

YOLO_ONNX_OUT="${YOLO_ONNX_OUT:-${REPO_ROOT}/artifacts/models/yolo_plates.onnx}"
UPDATE_CONF="${UPDATE_CONF:-0}"
PREFIX="${PREFIX:-/usr/local}"
BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}"

require_cmd cmake
require_cmd python3

log "Using CONFIG_DIR=${CONFIG_DIR}"
log "Using YOLO_PT=${YOLO_PT}"
log "ONNX output will be ${YOLO_ONNX_OUT}"
log "UPDATE_CONF=${UPDATE_CONF}"

mkdir -p "$(dirname "${YOLO_ONNX_OUT}")"

log "Exporting YOLO to ONNX"
python3 "${SCRIPT_DIR}/export_yolo.py" --pt "${YOLO_PT}" --out "${YOLO_ONNX_OUT}"

if [[ "${UPDATE_CONF}" == "1" ]]; then
  CONF_FILE=""
  if [[ -f "${CONFIG_DIR}/openalpr.conf" ]]; then
    CONF_FILE="${CONFIG_DIR}/openalpr.conf"
  elif [[ -f "${CONFIG_DIR}/openalpr.conf.defaults" ]]; then
    CONF_FILE="${CONFIG_DIR}/openalpr.conf.defaults"
  else
    die "No openalpr.conf or openalpr.conf.defaults in ${CONFIG_DIR}"
  fi
  log "Updating yolo_model_path in ${CONF_FILE}"
  python3 - "$CONF_FILE" "${YOLO_ONNX_OUT}" <<'PY'
import json, os, sys, tempfile
conf_path = sys.argv[1]
new_path = sys.argv[2]
lines = []
with open(conf_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
updated = False
out_lines = []
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("#") or stripped.strip() == "":
        out_lines.append(line)
        continue
    if stripped.lower().startswith("yolo_model_path"):
        prefix = line.split("=", 1)[0]
        out_lines.append(f"{prefix.strip()} = {new_path}\n")
        updated = True
    else:
        out_lines.append(line)
if not updated:
    out_lines.append(f"yolo_model_path = {new_path}\n")
tmp_path = conf_path + ".tmp"
target_path = conf_path
try:
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    os.replace(tmp_path, target_path)
    print(f"[run_all][conf] updated {target_path}")
except PermissionError:
    alt = conf_path + ".new"
    with open(alt, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"[run_all][conf] insufficient permission; wrote {alt}")
PY
fi

find_alpr() {
  for cand in \
    "${PREFIX}/bin/alpr" \
    "${BUILD_DIR}/alpr" \
    "${BUILD_DIR}/src/alpr" \
    "${REPO_ROOT}/${BUILD_DIR}/alpr" \
    "${REPO_ROOT}/${BUILD_DIR}/src/alpr" \
    "$(command -v alpr 2>/dev/null || true)"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

ALPR_BIN="$(find_alpr || true)"
if [[ -z "${ALPR_BIN}" ]]; then
  log "alpr not found; configuring and building"
  cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${BUILD_DIR}" -j"${JOBS}"
  ALPR_BIN="$(find_alpr || true)"
fi
[[ -n "${ALPR_BIN}" ]] || die "alpr binary not found after build"
log "Using ALPR_BIN=${ALPR_BIN}"

SMOKE_JSON="$(mktemp)"
log "Running smoke test JSON"
"${ALPR_BIN}" -c br -j "${TEST_IMAGE}" > "${SMOKE_JSON}"
python3 "${SCRIPT_DIR}/validate_json.py" "${SMOKE_JSON}"
log "Smoke test OK"

if command -v alpr-tool >/dev/null 2>&1; then
  log "Running alpr-tool export-yolo check"
  TOOL_OUT="$(dirname "${YOLO_ONNX_OUT}")/alpr_tool_test.onnx"
  alpr-tool export-yolo --model "${YOLO_PT}" --out "${TOOL_OUT}" --imgsz 640 || die "alpr-tool export-yolo failed"
  [[ -s "${TOOL_OUT}" ]] || die "alpr-tool export output empty: ${TOOL_OUT}"
fi

if command -v alprkit-smoke >/dev/null 2>&1; then
  log "Running alprkit-smoke"
  alprkit-smoke || die "alprkit-smoke failed"
fi

log "All steps completed successfully"

