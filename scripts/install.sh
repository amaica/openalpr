#!/usr/bin/env bash
set -euo pipefail

log() { echo "[alpr-install] $*"; }
err() { echo "[alpr-install][error] $*" >&2; exit 1; }

# Env defaults
PREFIX=${PREFIX:-/usr/local}
BUILD_DIR=${BUILD_DIR:-build}
JOBS=${JOBS:-$(nproc)}
INSTALL_DEPS=${INSTALL_DEPS:-1}
RUN_TESTS=${RUN_TESTS:-1}

# OS detection
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
else
  err "/etc/os-release not found; cannot detect distro."
fi

is_debian_like=false
if [[ "${ID_LIKE:-}" == *debian* || "${ID:-}" == "debian" || "${ID:-}" == "ubuntu" ]]; then
  is_debian_like=true
fi
$is_debian_like || err "Unsupported distro (requires Debian/Ubuntu). Detected ID=${ID:-unknown} ID_LIKE=${ID_LIKE:-unknown}"

SUDO=""
if [[ $(id -u) -ne 0 ]]; then
  SUDO="sudo"
fi

if [[ "$INSTALL_DEPS" != "0" ]]; then
  log "Installing build dependencies (non-interactive)"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  $SUDO apt-get install -y \
    build-essential cmake pkg-config git \
    libopencv-dev libtesseract-dev libleptonica-dev \
    libcurl4-openssl-dev liblog4cplus-dev libeigen3-dev \
    beanstalkd
fi

log "Configuring CMake (prefix=${PREFIX}, build dir=${BUILD_DIR})"
cmake -S src -B "${BUILD_DIR}" -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=RelWithDebInfo

log "Building (jobs=${JOBS})"
cmake --build "${BUILD_DIR}" -j"${JOBS}"

log "Installing to ${PREFIX}"
if [[ -n "$SUDO" ]]; then
  $SUDO cmake --install "${BUILD_DIR}"
  $SUDO install -m 0755 scripts/alpr-config.sh "${PREFIX}/bin/alpr-config"
else
  cmake --install "${BUILD_DIR}"
  install -m 0755 scripts/alpr-config.sh "${PREFIX}/bin/alpr-config"
fi

if [[ "${RUN_TESTS}" != "0" ]]; then
  [[ -n "${TEST_IMAGE:-}" ]] || err "TEST_IMAGE is required when RUN_TESTS=1"
  [[ -f "${TEST_IMAGE}" ]] || err "TEST_IMAGE not found: ${TEST_IMAGE}"

  ALPR_BIN="${PREFIX}/bin/alpr"
  if [[ ! -x "${ALPR_BIN}" ]]; then
    ALPR_BIN="${BUILD_DIR}/alpr"
  fi
  [[ -x "${ALPR_BIN}" ]] || err "alpr binary not found (looked in ${PREFIX}/bin and ${BUILD_DIR})"

  log "Running smoke test with ${ALPR_BIN}"
  SMOKE_LOG="$(mktemp)"
  if ! (cd "$(pwd)" && "${ALPR_BIN}" --config "$PWD/config/openalpr.conf.defaults" --runtime-dir "$PWD/runtime_data" -c us "${TEST_IMAGE}") >"${SMOKE_LOG}" 2>&1; then
    cat "${SMOKE_LOG}" >&2
    err "Smoke test failed. See log above."
  fi
  rm -f "${SMOKE_LOG}"
  log "Smoke test passed"
fi

log "Installation completed successfully."

