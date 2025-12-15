😂 agora ficou claríssimo.
Você quer **UM BLOCO ÚNICO**, **COMPLETO**, **FUNCIONAL**, **bilíngue**, **sem conversa**, **copiar–colar–acabou**.

Aqui está.
👉 **Copie EXATAMENTE o bloco abaixo e cole no `README.md`**.

---

```md
# 🚘 OpenALPR 2025 — Brazil & Mercosur (Cars & Motorcycles)

Modernized OpenALPR engine with **native Brazil/Mercosur support**, **real motorcycle plate recognition**, and **YOLOv8-based detection**, designed for **production-grade performance in C++**.

> 🇺🇸 English  
> 🇧🇷 Português abaixo

---

## 🇺🇸 English

### Overview
This project is a **deep, production-oriented evolution** of the classic OpenALPR engine.  
It preserves the original OCR pipeline while replacing outdated detection and country handling with a **modern, scalable, and configurable architecture**.

### Key Features
- Native support for **Brazil old plates (LLLNNNN)** and **Mercosur plates (LLLNLNN)**
- Explicit hybrid pipeline: **br2 → br** (no dependency on `eu/ad`)
- **Real motorcycle plate support**, not just detection
- YOLOv8 ONNX as primary detector (plugable by config)
- Automatic CPU/GPU backend detection (no flags, no recompilation)
- Safe fallback to classic detector
- Process-based parallelism (no shared state, no race conditions)

### Motorcycle Plates
- YOLO detects motorcycle plates reliably
- Dedicated OCR profiles:
  - `br_moto.conf`
  - `br2_moto.conf`
- Automatic profile selection using:
  - YOLO class (`plate_car` / `plate_moto`), or
  - bounding box aspect ratio
- Same validation rules (7 characters), tuned layout

### Architecture
```

Image / Frame
|
v
YOLO Detector (car / moto)
|
v
Vehicle Profile Selector
|-- br2 / br
|-- br2_moto / br_moto
|
v
OpenALPR OCR Pipeline
|
v
Pattern Validation + Fallback
|
v
Result (CLI / JSON / API)

````

### Configuration Example
```ini
detector_type = auto
yolo_model_path = /etc/openalpr/models/yolov8n_plates.onnx

br_enable_hybrid = 1
br_hybrid_order = br2,br
br_hybrid_min_confidence = 80

vehicle_profile_mode = auto
moto_aspect_ratio_min = 0.6
moto_aspect_ratio_max = 1.4
````

### Build

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Usage

```bash
alpr -c br car.jpg
alpr -c br motorcycle.jpg
```

---

## 🇧🇷 Português

### Visão Geral

Este projeto é uma **evolução profunda e voltada à produção** do OpenALPR clássico.
Ele mantém o pipeline de OCR original, substituindo apenas o que envelheceu, com foco em **Brasil, Mercosul, motos e performance real**.

### Principais Recursos

* Suporte nativo a placas **antigas** e **Mercosul**
* Pipeline híbrido explícito: **br2 → br**
* **Leitura real de placas de moto**, não apenas detecção
* Detector moderno com **YOLOv8 ONNX**
* Detecção automática de CPU/GPU
* Fallback seguro para detector clássico
* Paralelismo por processos (seguro e escalável)

### Placas de Moto

* Detecção via YOLO
* Perfis OCR dedicados:

  * `br_moto.conf`
  * `br2_moto.conf`
* Seleção automática do perfil por classe ou proporção
* OCR ajustado para layout de moto

### Build

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Uso

```bash
alpr -c br carro.jpg
alpr -c br moto.jpg
```

---

## Disclaimer / Aviso Legal

This project is open source and **not officially affiliated** with OpenALPR Inc.
Projeto open source **sem afiliação oficial** à OpenALPR Inc.

```

