#!/usr/bin/env bash
set -euo pipefail

# Garage-oriented wrapper:
# - Run ALPR on the original image
# - If best candidate is OLD BR format (@@@#### = LLLNNNN), DO NOT enhance
# - Otherwise, enhance with mini_enhancer --alpr-max and re-run ALPR
#
# Prints the final best plate, confidence, and which image was used.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ALPR_BIN="${ALPR_BIN:-$ROOT/build/src/alpr}"
ALPR_CFG="${ALPR_CFG:-$ROOT/config/openalpr.conf.defaults}"
ENH_BIN="${ENH_BIN:-$ROOT/mini_enhancer/build/mini_enhancer}"
COUNTRY="${COUNTRY:-br}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

usage() {
  cat <<'EOF'
Usage:
  ./garage_alpr.sh /path/to/image.png

Env overrides:
  ALPR_BIN=/path/to/alpr
  ALPR_CFG=/path/to/openalpr.conf
  ENH_BIN=/path/to/mini_enhancer
  COUNTRY=br
  MAX_PARALLEL=2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "" ]]; then
  usage
  exit 1
fi

img="$1"
if [[ ! -f "$img" ]]; then
  echo "ERROR: file not found: $img" >&2
  exit 2
fi

lock_fd=""
lock_slot=""
tmpdir=""
cleanup() {
  # release lock by closing fd
  if [[ -n "${lock_fd:-}" ]]; then
    eval "exec ${lock_fd}>&-"
  fi
  if [[ -n "${tmpdir:-}" && -d "${tmpdir:-}" ]]; then
    rm -rf -- "$tmpdir"
  fi
}
trap cleanup EXIT INT TERM

# Limit parallel executions to avoid CPU overload on weak machines.
acquire_slot() {
  local n="${MAX_PARALLEL}"
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then n=2; fi
  if (( n < 1 )); then n=1; fi
  if (( n > 4 )); then n=4; fi  # keep sane; machine is low-end

  while :; do
    for ((i=0; i<n; i++)); do
      local lf="/tmp/garage_alpr.slot.${i}.lock"
      # allocate fd dynamically
      local fd
      exec {fd}<> "$lf"
      if flock -n "$fd"; then
        lock_fd="$fd"
        lock_slot="$i"
        return 0
      fi
      # couldn't lock: close fd and try next
      eval "exec ${fd}>&-"
    done
    sleep 0.05
  done
}

acquire_slot

run_alpr() {
  "$ALPR_BIN" -c "$COUNTRY" --config "$ALPR_CFG" "$1"
}

# Extracts first candidate and its confidence from alpr output.
extract_best() {
  # Expected line format:
  #     - ABC1234     confidence: 88.2494
  run_alpr "$1" | awk '
    $1=="-" {
      plate=$2;
      conf=0;
      for (i=1;i<=NF;i++) if ($i=="confidence:") { conf=$(i+1); break }
      print plate, conf;
      exit
    }'
}

best_line="$(extract_best "$img" || true)"
best_plate="$(awk '{print $1}' <<<"$best_line")"
best_conf="$(awk '{print $2}' <<<"$best_line")"

# If no plate found, force enhancement attempt.
need_enhance=1
if [[ -n "${best_plate:-}" ]]; then
  # Old BR template: @@@#### (LLLNNNN)
  if [[ "$best_plate" =~ ^[A-Z]{3}[0-9]{4}$ ]]; then
    need_enhance=0
  fi
fi

used_img="$img"
if [[ "$need_enhance" -eq 1 ]]; then
  # Avoid output collisions across concurrent requests:
  # copy input to a unique temp dir and run enhancer there.
  tmpdir="$(mktemp -d /tmp/garage_alpr.XXXXXX)"
  inbase="$(basename -- "$img")"
  tmp_in="$tmpdir/$inbase"
  cp -f -- "$img" "$tmp_in"

  enh_out="$("$ENH_BIN" --alpr-max "$tmp_in" | awk '/^Done: /{print $2; exit}')"
  if [[ -z "${enh_out:-}" || ! -f "$enh_out" ]]; then
    echo "ERROR: enhancement failed for: $img" >&2
    exit 3
  fi
  used_img="$enh_out"
  best_line="$(extract_best "$used_img" || true)"
  best_plate="$(awk '{print $1}' <<<"$best_line")"
  best_conf="$(awk '{print $2}' <<<"$best_line")"
fi

if [[ -z "${best_plate:-}" ]]; then
  echo "NO_PLATE used=$used_img" >&2
  exit 4
fi

echo "PLATE=$best_plate CONF=$best_conf USED=$used_img"

