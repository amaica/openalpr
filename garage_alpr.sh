#!/usr/bin/env bash
set -euo pipefail

# Garage-oriented wrapper:
# - Run ALPR on the original image
# - If best candidate is OLD BR format (@@@#### = LLLNNNN), DO NOT enhance
# - Otherwise, enhance with mini_enhancer --alpr-max and re-run ALPR
#
# Human stdout (default): PLATE=… CONF=… USED=…
# Machine stdout (--json): one JSON line; errors also one JSON line on stderr when --json
#
# Exit codes (stable for integradores):
#   0  success (plate read)
#   1  missing image argument (usage printed)
#   2  input file not found
#   3  enhancement pipeline failed (mini_enhancer)
#   4  no plate after original (and enhanced if applicable)
#   5  required binary missing or not executable

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ALPR_BIN="${ALPR_BIN:-$ROOT/build/src/alpr}"
ALPR_CFG="${ALPR_CFG:-$ROOT/config/openalpr.conf.defaults}"
ENH_BIN="${ENH_BIN:-$ROOT/mini_enhancer/build/mini_enhancer}"
COUNTRY="${COUNTRY:-br}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

JSON_OUT=0

usage() {
  cat <<'EOF'
Usage:
  ./garage_alpr.sh [--json] /path/to/image.png

Options:
  --json   Print one JSON object on stdout (success); on failure print JSON on stderr

Exit codes:
  0 success  1 missing arg  2 file not found  3 enhancer failed  4 no plate  5 missing binary

Env overrides:
  ALPR_BIN=/path/to/alpr
  ALPR_CFG=/path/to/openalpr.conf
  ENH_BIN=/path/to/mini_enhancer
  COUNTRY=br
  MAX_PARALLEL=2
EOF
}

json_err() {
  local reason="$1" code="$2" path="${3:-}"
  python3 -c 'import json,sys; print(json.dumps({"error":sys.argv[1],"code":int(sys.argv[2]),"processed_path":sys.argv[3]}), file=sys.stderr, flush=True)' \
    "$reason" "$code" "$path" || echo "{\"error\":\"${reason}\",\"code\":${code},\"processed_path\":\"${path}\"}" >&2
}

while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --json) JSON_OUT=1; shift ;;
    *) usage >&2; exit 1 ;;
  esac
done

img="${1:-}"
if [[ -z "$img" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$img" ]]; then
  if [[ "$JSON_OUT" -eq 1 ]]; then json_err "FILE_NOT_FOUND" 2 "$img"; else echo "ERROR: file not found: $img" >&2; fi
  exit 2
fi

if [[ ! -x "$ALPR_BIN" ]]; then
  if [[ "$JSON_OUT" -eq 1 ]]; then json_err "ALPR_BIN_NOT_EXECUTABLE" 5 "$ALPR_BIN"; else echo "ERROR: not executable: $ALPR_BIN" >&2; fi
  exit 5
fi
if [[ ! -x "$ENH_BIN" ]]; then
  if [[ "$JSON_OUT" -eq 1 ]]; then json_err "ENH_BIN_NOT_EXECUTABLE" 5 "$ENH_BIN"; else echo "ERROR: not executable: $ENH_BIN" >&2; fi
  exit 5
fi

lock_fd=""
lock_slot=""
tmpdir=""
cleanup() {
  if [[ -n "${lock_fd:-}" ]]; then
    eval "exec ${lock_fd}>&-"
  fi
  if [[ -n "${tmpdir:-}" && -d "${tmpdir:-}" ]]; then
    rm -rf -- "$tmpdir"
  fi
}
trap cleanup EXIT INT TERM

acquire_slot() {
  local n="${MAX_PARALLEL}"
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then n=2; fi
  if (( n < 1 )); then n=1; fi
  if (( n > 4 )); then n=4; fi

  while :; do
    for ((i=0; i<n; i++)); do
      local lf="/tmp/garage_alpr.slot.${i}.lock"
      local fd
      exec {fd}<> "$lf"
      if flock -n "$fd"; then
        lock_fd="$fd"
        lock_slot="$i"
        return 0
      fi
      eval "exec ${fd}>&-"
    done
    sleep 0.05
  done
}

acquire_slot

run_alpr() {
  "$ALPR_BIN" -c "$COUNTRY" --config "$ALPR_CFG" "$1"
}

extract_best() {
  run_alpr "$1" | awk '
    $1=="-" {
      plate=$2;
      conf=0;
      for (i=1;i<=NF;i++) if ($i=="confidence:") { conf=$(i+1); break }
      print plate, conf;
      exit
    }'
}

json_ok() {
  local plate="$1" conf="$2" used_mode="$3" processed="$4" source="$5"
  python3 -c 'import json,sys
c=sys.argv[2]
try: cf=float(c) if c else 0.0
except ValueError: cf=0.0
print(json.dumps({"plate":sys.argv[1],"confidence":cf,"used":sys.argv[3],"processed_path":sys.argv[4],"source_path":sys.argv[5]}))' \
    "$plate" "$conf" "$used_mode" "$processed" "$source"
}

best_line="$(extract_best "$img" || true)"
best_plate="$(awk '{print $1}' <<<"$best_line")"
best_conf="$(awk '{print $2}' <<<"$best_line")"

need_enhance=1
if [[ -n "${best_plate:-}" ]]; then
  if [[ "$best_plate" =~ ^[A-Z]{3}[0-9]{4}$ ]]; then
    need_enhance=0
  fi
fi

used_img="$img"
used_mode="original"
if [[ "$need_enhance" -eq 1 ]]; then
  tmpdir="$(mktemp -d /tmp/garage_alpr.XXXXXX)"
  inbase="$(basename -- "$img")"
  tmp_in="$tmpdir/$inbase"
  cp -f -- "$img" "$tmp_in"

  enh_out="$("$ENH_BIN" --alpr-max "$tmp_in" | awk '/^Done: /{print $2; exit}')"
  if [[ -z "${enh_out:-}" || ! -f "$enh_out" ]]; then
    if [[ "$JSON_OUT" -eq 1 ]]; then json_err "ENHANCEMENT_FAILED" 3 "$img"; else echo "ERROR: enhancement failed for: $img" >&2; fi
    exit 3
  fi
  used_img="$enh_out"
  used_mode="enhanced"
  best_line="$(extract_best "$used_img" || true)"
  best_plate="$(awk '{print $1}' <<<"$best_line")"
  best_conf="$(awk '{print $2}' <<<"$best_line")"
fi

if [[ -z "${best_plate:-}" ]]; then
  if [[ "$JSON_OUT" -eq 1 ]]; then json_err "NO_PLATE" 4 "$used_img"; else echo "NO_PLATE used=$used_img" >&2; fi
  exit 4
fi

if [[ "$JSON_OUT" -eq 1 ]]; then
  json_ok "$best_plate" "$best_conf" "$used_mode" "$used_img" "$img"
else
  echo "PLATE=$best_plate CONF=$best_conf USED=$used_img"
fi
