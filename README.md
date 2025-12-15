# 🚘 OpenALPR 2025 — Brazil & Mercosur
### Cars & Motorcycles • YOLOv8 • Production-grade C++

Modernized OpenALPR engine with **native Brazil/Mercosur support**, **real motorcycle OCR**, and **YOLOv8-based detection**, designed for **high-performance, scalable C++ production systems**.

---

## 🇺🇸 English

## Overview
This project is a **production-grade architectural evolution** of the classic OpenALPR engine.  
It **preserves the proven OCR pipeline** while replacing outdated detection, country handling, and scalability limitations with a **modern, configurable, and robust architecture**.

## Core Capabilities

### Brazil & Mercosur (Native)
- Old Brazilian plates: **LLLNNNN**
- Mercosur plates: **LLLNLNN**
- Native hybrid pipeline: **br2 → br**
- Explicit, deterministic, and logged fallback rules

### Motorcycle Plates (Real Support)
- YOLOv8 detects motorcycle plates reliably
- Dedicated OCR profiles:
  - `br_moto.conf`
  - `br2_moto.conf`
- Automatic vehicle-type selection using:
  - YOLO class (`plate_car` / `plate_moto`)
  - Bounding box aspect ratio (fallback)
- Same 7-character validation rules with tuned segmentation/layout

### Modern Detection
- YOLOv8 exported to **ONNX**
- Model loaded dynamically via configuration
- No recompilation required to update models
- Automatic backend selection (CPU / CUDA when available)
- Safe fallback to classical detector

### Performance & Scalability
- Process-based parallelism (not threads)
- One YOLO + one ALPR instance per worker
- Linear scaling with CPU/GPU
- No shared state, no race conditions
- Suitable for batch processing and video streams

## Architecture
```
Input (Image / Video Frame)
        |
        v
+----------------------+
|  YOLOv8 Detector    |
| (car / motorcycle)  |
+----------------------+
        |
        v
+----------------------+
| Vehicle Type Selector|
+----------------------+
        |
        v
+----------------------+
| OCR Profile Selector |
| br2 / br             |
| br2_moto / br_moto   |
+----------------------+
        |
        v
+----------------------+
| OpenALPR OCR Engine  |
+----------------------+
        |
        v
+----------------------+
| Pattern Validation   |
| + Explicit Fallback  |
+----------------------+
        |
        v
Output (CLI / JSON / API)
```

## Configuration Example
```ini
detector_type = auto
yolo_model_path = /etc/openalpr/models/yolov8n_plates.onnx

br_enable_hybrid = 1
br_hybrid_order = br2,br
br_hybrid_min_confidence = 80

vehicle_profile_mode = auto
moto_aspect_ratio_min = 0.6
moto_aspect_ratio_max = 1.4
```

## Build
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

Install (one-liner, non-interativo):
```bash
sudo TEST_IMAGE=/caminho/para/imagem.jpg ./scripts/install.sh
```

## Instalação automática (Linux Debian/Ubuntu)
- Pré-requisitos: Linux Debian/Ubuntu com sudo
- Comando único:
  ```bash
  sudo TEST_IMAGE=/caminho/para/imagem.jpg ./scripts/install.sh
  ```
- O script:
  - Detecta a distro
  - Instala dependências via apt (non-interactive)
  - Configura e compila com CMake
  - Instala os binários
  - Executa smoke test com a imagem indicada
- Variáveis de ambiente suportadas:
  - `PREFIX` (padrão: /usr/local)
  - `BUILD_DIR` (padrão: build)
  - `JOBS` (padrão: nproc)
  - `INSTALL_DEPS` (padrão: 1)
  - `RUN_TESTS` (padrão: 1)
  - `TEST_IMAGE` (obrigatória para o smoke test)

## Usage
```bash
alpr -c br car.jpg
alpr -c br motorcycle.jpg
```

## Disclaimer
This project is open source and **not officially affiliated** with OpenALPR Inc.

---

## 🇧🇷 Português

## Visão Geral
Este projeto é uma **evolução arquitetural de nível produção** do OpenALPR clássico.  
Ele **mantém o pipeline de OCR consolidado** e substitui os componentes obsoletos por uma **arquitetura moderna, configurável e robusta**.

## Capacidades Principais

### Brasil e Mercosul (Nativo)
- Placas antigas: **LLLNNNN**
- Placas Mercosul: **LLLNLNN**
- Pipeline híbrido explícito: **br2 → br**
- Fallback determinístico e logado

### Placas de Moto (Suporte Real)
- Detecção confiável com YOLOv8
- Perfis OCR dedicados:
  - `br_moto.conf`
  - `br2_moto.conf`
- Seleção automática por:
  - classe do YOLO (`plate_car` / `plate_moto`)
  - proporção da bounding box (fallback)
- Validação de 7 caracteres com layout ajustado

### Detecção Moderna
- YOLOv8 em **ONNX**
- Modelo carregado via configuração
- Sem recompilação para atualizar modelos
- Seleção automática de backend (CPU / CUDA)
- Fallback seguro para detector clássico

### Performance e Escalabilidade
- Paralelismo por processos
- Um YOLO + um ALPR por worker
- Escala linear com CPU/GPU
- Sem estado compartilhado

## Build e Uso
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)

alpr -c br carro.jpg
alpr -c br moto.jpg
```

## Aviso Legal
Projeto open source, **sem afiliação oficial** com a OpenALPR Inc.
