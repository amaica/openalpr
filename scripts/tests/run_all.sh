#!/usr/bin/env bash
set -euo pipefail

log() { echo "[run_all] $*"; }
die() { echo "[run_all][error] $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ARTIFACTS_ROOT="${REPO_ROOT}/artifacts"
LOG_DIR="${ARTIFACTS_ROOT}/logs"
MODEL_DIR="${ARTIFACTS_ROOT}/models"
RESULTS_DIR="${ARTIFACTS_ROOT}/results"
mkdir -p "${LOG_DIR}" "${MODEL_DIR}" "${RESULTS_DIR}"

TEST_IMAGE="${TEST_IMAGE:-}"
[[ -n "${TEST_IMAGE}" ]] || die "TEST_IMAGE is required (path to an image for smoke test)"
[[ -f "${TEST_IMAGE}" ]] || die "TEST_IMAGE not found: ${TEST_IMAGE}"

YOLO_PT="${YOLO_PT:-/home/aurelio/FONTES/python/SynkiSentinel/models/plates/yolov8n_plates.pt}"
[[ -f "${YOLO_PT}" ]] || die "YOLO_PT not found: ${YOLO_PT}"

YOLO_ONNX_OUT="${YOLO_ONNX_OUT:-${MODEL_DIR}/yolo_plates.onnx}"
UPDATE_CONF="${UPDATE_CONF:-0}"
PREFIX="${PREFIX:-/usr/local}"
BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}"

require_cmd cmake
require_cmd python3

STEP_SUMMARY=()
FAIL_FLAG=0
FAIL_DETAIL=""

record() {
  local status="$1"; shift
  local name="$1"; shift
  local info="$1"; shift
  STEP_SUMMARY+=("[$status] ${name} -> ${info}")
  if [[ "${status}" == "FAIL" && "${FAIL_FLAG}" -eq 0 ]]; then
    FAIL_FLAG=1
    FAIL_DETAIL="${info}"
  fi
}

run_step() {
  local name="$1"; shift
  local log_file="$1"; shift
  local cmd_str="$*"
  set +e
  eval "${cmd_str}" >"${log_file}" 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    record "PASS" "${name}" "${log_file}"
  else
    record "FAIL" "${name}" "${log_file}"
    echo "[FAIL] ${name}"
    echo "Command: ${cmd_str}"
    echo "Log: ${log_file}"
    exit 1
  fi
}

CONFIG_DIR="${CONFIG_DIR:-}"

log "Using YOLO_PT=${YOLO_PT}"
log "ONNX output will be ${YOLO_ONNX_OUT}"
log "UPDATE_CONF=${UPDATE_CONF}"

run_step "export_yolo" "${LOG_DIR}/export_yolo.log" \
  "python3 \"${SCRIPT_DIR}/export_yolo.py\" --pt \"${YOLO_PT}\" --out \"${YOLO_ONNX_OUT}\""

if [[ "${UPDATE_CONF}" == "1" ]]; then
  if [[ -z "${CONFIG_DIR}" ]]; then
    record "SKIP" "update_conf" "CONFIG_DIR not set; skipping"
  else
    CONF_FILE=""
    if [[ -f "${CONFIG_DIR}/openalpr.conf" ]]; then
      CONF_FILE="${CONFIG_DIR}/openalpr.conf"
    elif [[ -f "${CONFIG_DIR}/openalpr.conf.defaults" ]]; then
      CONF_FILE="${CONFIG_DIR}/openalpr.conf.defaults"
    else
      record "FAIL" "update_conf" "No conf in ${CONFIG_DIR}"
      echo "[FAIL] update_conf"
      echo "Reason: no openalpr.conf/openalpr.conf.defaults in ${CONFIG_DIR}"
      exit 1
    fi
    python3 - "${CONF_FILE}" "${YOLO_ONNX_OUT}" > "${LOG_DIR}/update_conf.log" 2>&1 <<'PY'
import os, sys
conf_path = sys.argv[1]
new_path = sys.argv[2]
with open(conf_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
out_lines = []
updated = False
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
try:
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    os.replace(tmp_path, conf_path)
    print(f"[run_all][conf] updated {conf_path}")
except PermissionError:
    alt = conf_path + ".new"
    with open(alt, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"[run_all][conf] insufficient permission; wrote {alt}")
    print(alt)
PY
    record "PASS" "update_conf" "${LOG_DIR}/update_conf.log"
  fi
else
  record "SKIP" "update_conf" "UPDATE_CONF=0"
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
  run_step "cmake_build" "${LOG_DIR}/cmake_build.log" \
    "cmake -S \"${REPO_ROOT}\" -B \"${BUILD_DIR}\" -DCMAKE_BUILD_TYPE=Release && cmake --build \"${BUILD_DIR}\" -j\"${JOBS}\""
  ALPR_BIN="$(find_alpr || true)"
  [[ -n "${ALPR_BIN}" ]] || die "alpr binary not found after build (see ${LOG_DIR}/cmake_build.log)"
else
  record "SKIP" "cmake_build" "alpr present at ${ALPR_BIN}"
fi
log "Using ALPR_BIN=${ALPR_BIN}"

ALPR_JSON="${RESULTS_DIR}/alpr_output.json"
run_step "alpr_smoke" "${LOG_DIR}/alpr_smoke.log" \
  "\"${ALPR_BIN}\" -c br -j \"${TEST_IMAGE}\" > \"${ALPR_JSON}\""

run_step "json_validate" "${LOG_DIR}/json_validate.log" \
  "python3 \"${SCRIPT_DIR}/validate_json.py\" \"${ALPR_JSON}\""

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
for entry in "${STEP_SUMMARY[@]}"; do
  case "${entry}" in
    "[PASS]"*) ((PASS_COUNT++)) ;;
    "[FAIL]"*) ((FAIL_COUNT++)) ;;
    "[SKIP]"*) ((SKIP_COUNT++)) ;;
  esac
done

echo "================ TEST REPORT ================"
for entry in "${STEP_SUMMARY[@]}"; do
  echo "${entry}"
done
echo "---------------------------------------------"
if [[ ${FAIL_COUNT} -eq 0 ]]; then
  echo "RESULT: PASS (${PASS_COUNT}/${#STEP_SUMMARY[@]})"
else
  echo "RESULT: FAIL"
  echo "Failed step logged at: ${FAIL_DETAIL}"
fi
echo "Artifacts: ${ARTIFACTS_ROOT}"
echo "============================================="

