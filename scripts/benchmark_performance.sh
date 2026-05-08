#!/usr/bin/env bash
# Benchmarks reprodutíveis: alpr sequencial vs --jobs, garage_alpr, mini_enhancer.
# GNU /usr/bin/time -f '%e' = segundos (wall clock). Na raiz: ./scripts/benchmark_performance.sh
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

ALPR="${ALPR:-$ROOT/build/src/alpr}"
CFG="${CFG:-$ROOT/config/openalpr.conf.defaults}"
GARAGE="${GARAGE:-$ROOT/garage_alpr.sh}"
ENH="${ENH:-$ROOT/mini_enhancer/build/mini_enhancer}"
REPEAT="${REPEAT:-5}"

tim() {
  local label="$1"
  shift
  local tf sec
  tf="$(mktemp)"
  /usr/bin/time -o "$tf" -f '%e' "$@" >/dev/null 2>&1 || true
  sec="$(tr -d '\n' < "$tf")"
  rm -f "$tf"
  printf '  %-52s %8s s\n' "$label" "$sec"
}

processing_ms() {
  local f="$1"
  "$ALPR" -c br --config "$CFG" -j "$f" 2>/dev/null | tail -1 | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('processing_time_ms',-1))"
}

echo "=============================================="
echo " OpenALPR / garage — benchmarks"
echo " ROOT=$ROOT  REPEAT=$REPEAT"
echo " $(date -Iseconds)"
echo "=============================================="
echo

echo "## Testes automáticos"
if [[ -f "$ROOT/python/test_garage_alpr_client.py" ]]; then
  tim "python3 python/test_garage_alpr_client.py" python3 "$ROOT/python/test_garage_alpr_client.py"
fi
if [[ -x "$ROOT/scripts/test_garage_alpr_contract.sh" ]]; then
  tim "bash scripts/test_garage_alpr_contract.sh" bash "$ROOT/scripts/test_garage_alpr_contract.sh"
fi
UT="$(find "$ROOT/build" -name unittests -type f -executable 2>/dev/null | head -1)"
if [[ -n "${UT:-}" ]]; then
  tim "C++ unittests ($UT)" "$UT"
else
  echo "  C++ unittests: SKIP (executable unittests not found under build/)"
fi
echo

if [[ ! -x "$ALPR" ]]; then
  echo "ERROR: missing $ALPR" >&2
  exit 1
fi

echo "## alpr — wall clock (segundos)"
export A="$ALPR" C="$CFG" R="$REPEAT"
for img in 1.png 2.png; do
  [[ -f "$ROOT/$img" ]] || continue
  export P="$ROOT/$img"
  tim "alpr 1× $img" "$A" -c br --config "$C" "$P"
  tim "alpr ${REPEAT}× sequencial $img" bash -c 'for ((i=1;i<=R;i++)); do "$A" -c br --config "$C" "$P" >/dev/null; done'
  args=()
  for ((i=1;i<=R;i++)); do args+=("$P"); done
  tim "alpr 1 invocação --jobs=2 (${REPEAT}× mesmo ficheiro)" "$A" -c br --config "$C" --jobs 2 "${args[@]}"
done
unset A C R P
echo

echo "## alpr — processing_time_ms (JSON, uma passagem)"
for img in 1.png 2.png; do
  [[ -f "$ROOT/$img" ]] || continue
  ms="$(processing_ms "$ROOT/$img")"
  echo "  $img  processing_time_ms=$ms"
done
echo

if [[ -x "$GARAGE" ]]; then
  echo "## garage_alpr.sh"
  export G="$GARAGE" R="$REPEAT"
  for img in 1.png 2.png; do
    [[ -f "$ROOT/$img" ]] || continue
    export P="$ROOT/$img"
    tim "garage_alpr ${REPEAT}× $img" bash -c 'for ((i=1;i<=R;i++)); do "$G" "$P" >/dev/null; done'
  done
  unset G R P
  echo
fi

if [[ -x "$ENH" ]]; then
  echo "## mini_enhancer --alpr-max (1×)"
  for img in 1.png 2.png; do
    [[ -f "$ROOT/$img" ]] || continue
    tim "mini_enhancer --alpr-max $img" "$ENH" --alpr-max "$ROOT/$img"
  done
  echo
fi

echo "## Notas"
echo "  - Tempos = tempo de parede do processo (/usr/bin/time)."
echo "  - alpr sequencial vs --jobs: na mesma invocação, workers evitam relançar tudo por ficheiro."
echo "  - garage_alpr inclui lógica enhance + segundo alpr quando aplicável."
echo "=============================================="
