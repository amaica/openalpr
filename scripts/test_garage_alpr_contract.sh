#!/usr/bin/env bash
# Contract tests for garage_alpr.sh (exit codes + --json). Run from repo root or via ctest.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
GARAGE="${ROOT}/garage_alpr.sh"
ALPR="${ROOT}/build/src/alpr"
ENH="${ROOT}/mini_enhancer/build/mini_enhancer"

die() { echo "FAIL: $*" >&2; exit 1; }

assert_eq() {
  local got="$1" want="$2" msg="$3"
  [[ "$got" == "$want" ]] || die "$msg (got=$got want=$want)"
}

[[ -x "$GARAGE" ]] || chmod +x "$GARAGE" 2>/dev/null || true
[[ -f "$GARAGE" ]] || die "missing $GARAGE"

# 1: missing image
set +e
out="$(bash "$GARAGE" 2>&1)"
st=$?
set -e
assert_eq "$st" 1 "missing arg should exit 1"

# 2: file not found
set +e
bash "$GARAGE" "${ROOT}/___no_such_image_$$.png" >/dev/null 2>&1
st=$?
set -e
assert_eq "$st" 2 "missing file should exit 2"

# 2 + JSON stderr
set +e
errf="/tmp/garage_contract_err_$$.txt"
bash "$GARAGE" --json "${ROOT}/___no_such_image_$$.png" >/dev/null 2>"$errf"
st=$?
jerr="$(cat "$errf")"
rm -f "$errf"
set -e
assert_eq "$st" 2 "missing file --json should exit 2"
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$jerr" >/dev/null || die "stderr should be JSON"

# 5: missing alpr binary (valid placeholder file so we pass file checks)
touch "${ROOT}/.garage_contract_dummy_$$"
trap 'rm -f "${ROOT}/.garage_contract_dummy_$$"' EXIT
set +e
ALPR_BIN="${ROOT}/__no_alpr_binary_$$" bash "$GARAGE" "${ROOT}/.garage_contract_dummy_$$" >/dev/null 2>&1
st=$?
set -e
assert_eq "$st" 5 "bad ALPR_BIN should exit 5"

# Optional: full pipeline if build artifacts exist (reference images: 1.png, 2.png only)
if [[ -x "$ALPR" && -x "$ENH" ]]; then
  for img in 1.png 2.png; do
    f="${ROOT}/${img}"
    if [[ ! -f "$f" ]]; then
      echo "SKIP: $f not present (optional integration check)"
      continue
    fi
    set +e
    line="$(bash "$GARAGE" "$f" 2>&1)"
    st=$?
    set -e
    assert_eq "$st" 0 "$img should succeed when present"
    [[ "$line" == PLATE=* ]] || die "expected PLATE= line for $img, got: $line"
    set +e
    js="$(bash "$GARAGE" --json "$f" 2>/dev/null)"
    st=$?
    set -e
    assert_eq "$st" 0 "$img --json should succeed"
    python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert "plate" in d and d["plate"]; assert d.get("used") in ("original","enhanced")' "$js" >/dev/null \
      || die "bad success JSON for $img: $js"
  done
else
  echo "SKIP: alpr or mini_enhancer not built (optional integration check)"
fi

echo "OK garage_alpr contract tests"
