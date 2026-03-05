#!/bin/bash
# Instala dependências e faz build do OpenALPR + configurador web.
# Execute na raiz do repo: ./scripts/install_and_build_all.sh
# Quando pedir senha sudo, digite a sua.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1. Instalando dependências (sudo) ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential cmake \
  libopencv-dev liblog4cplus-dev libtesseract-dev libleptonica-dev

echo "=== 2. Build OpenALPR (build/) ==="
mkdir -p build
cd build
cmake ..
make -j"$(nproc)" 2>/dev/null || make -j2
cd "$REPO_ROOT"

echo "=== 3. Build configurador web (web/build/) ==="
mkdir -p web/build
cd web/build
cmake ..
make -j"$(nproc)" 2>/dev/null || make -j2
cd "$REPO_ROOT"

echo "=== Pronto ==="
echo "  OpenALPR tool:  ./build/src/alpr-tool preview --country br --source /caminho/video.mp4"
echo "  Web configurator: ./web/build/bin/alpr-web-configurator  (porta 18080)"
echo "  Abra: http://localhost:18080"
