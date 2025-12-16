#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/src/tools/alpr-configurator/alpr-configurator"

if [[ ! -x "$BIN" ]]; then
  echo "[run-configurator] binary not found, building..."
  cmake --build "${ROOT}/build" -j"$(nproc)" --target alpr-configurator
fi

echo "Running: ${BIN}"
exec "${BIN}" "$@"

