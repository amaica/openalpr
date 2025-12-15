#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT_DIR}/third_party/onnxruntime"
VERSION="${ONNXRUNTIME_VERSION:-1.17.3}"

arch="$(uname -m)"
case "$arch" in
  x86_64)   PKG="onnxruntime-linux-x64-${VERSION}.tgz" ;;
  aarch64|arm64) PKG="onnxruntime-linux-aarch64-${VERSION}.tgz" ;;
  *)
    echo "Unsupported arch: $arch" >&2
    exit 1
    ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

echo "[fetch_onnxruntime] Downloading ${URL}"
curl -L "$URL" -o "$TMP"

rm -rf "$DEST"
mkdir -p "$DEST"
echo "[fetch_onnxruntime] Extracting to ${DEST}"
tar -xzf "$TMP" -C "$DEST" --strip-components=1

if [ ! -f "${DEST}/include/onnxruntime_cxx_api.h" ] || [ ! -f "${DEST}/lib/libonnxruntime.so" ]; then
  echo "[fetch_onnxruntime] ERROR: onnxruntime artifacts not found after extraction" >&2
  exit 1
fi

real_lib=$(readlink -f "${DEST}/lib/libonnxruntime.so")
size_lib=$(stat -c%s "${real_lib}")
if [ "${size_lib}" -lt 1000000 ]; then
  echo "[fetch_onnxruntime] ERROR: libonnxruntime.so size too small (${size_lib})" >&2
  exit 1
fi

echo "[fetch_onnxruntime] OK version=${VERSION} arch=${arch} lib_size=${size_lib} lib=${real_lib}"

