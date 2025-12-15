#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
alpr-config: wrapper for alpr-tool (roi/tune/preview/export-yolo)
Usage:
  alpr-config roi [--source ...] [--conf ...]
  alpr-config tune [--source ...] [--conf ...]
  alpr-config preview [--source ...] [--conf ...]
  alpr-config export-yolo --model <.pt> --out <.onnx> [--imgsz N] [--update-conf --conf ...]
EOF
  exit 0
fi

tool="alpr-tool"
if ! command -v "${tool}" >/dev/null 2>&1; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  for alt in "${script_dir}/../build/alpr-tool" "${script_dir}/../build/src/alpr-tool"; do
    if [[ -x "${alt}" ]]; then
      tool="${alt}"
      break
    fi
  done
  if [[ "${tool}" != "alpr-tool" && -x "${tool}" ]]; then
    :
  else
    echo "alpr-tool not found in PATH or in build outputs" >&2
    exit 127
  fi
fi

exec "${tool}" "$@"

