# OpenALPR 2025 — Brazil & Mercosur
### Cars & Motorcycles • Profiles • Speed-First OCR • C++

A production-oriented evolution of the classic OpenALPR C/C++ stack focused on **Brazil/Mercosur**, **operational reliability**, and **measurable performance**. This repository keeps dependencies minimal (classic detector + Tesseract OCR), adds **profile-based OCR strategies** (including motorcycle-friendly behavior), and exposes **metrics** so improvements are driven by numbers—not guesses.

> **No YOLO inside this library.** If you use YOLO/trackers, they live in the application layer and can feed bboxes into the ALPR pipeline via **skip-detection** scenarios.

---

## 🇺🇸 English

## What’s in here (2025-ready features)

### 1) Profile tag (NEW)
`alpr-tool preview` supports:

- `--profile=default|moto|garagem`

Profiles change the OCR strategy at runtime:

| Profile | Intent | Current behavior |
|---|---|---|
| `default` | cars / general | `ocr_burst_frames = 1` |
| `moto` | motorcycles / small plates | `ocr_burst_frames = 6` + temporal voting |
| `garagem` | garages / low-speed | `ocr_burst_frames = 10` + temporal voting |

✅ Implemented in: `src/tools/alpr_tool.cpp`  
✅ Commit: `feat(tool): add profile tag with burst OCR and temporal voting`

---

### 2) Temporal voting (NEW)
When `profile` is `moto` or `garagem`, the tool runs burst OCR and aggregates results via majority voting.

Example log:
```
[vote] profile=moto plate=ABC1D23 window=N
```

---

### 3) Report / Metrics (NEW)
The preview report and its JSON now include:

- `profile`
- `ocr_burst_frames`
- `votes_emitted`
- `final_plate_count`

These are designed to make performance and quality comparable across runs.

---

### 4) Minimal-dependency core (Architectural decision)
- Classic OpenALPR detector (no modern DNN detector embedded)
- Tesseract-based OCR in the core (C++ integration)
- **skip_detection** exists (off by default) for bbox-provided workflows

---

### 5) Java wrapper bounding boxes (NEW / Improved)
Recent work improved Java-side usability for plate coordinates and added proof output.

- “bbox proof” output exists in Java tests
- More ergonomic API exposure for plate box retrieval

(See latest Java commits in your repo history.)

---

## Quick start (CLI)

### Build
```bash
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
```

### Preview (fastest / default)
```bash
./build/src/alpr-tool preview \
  --profile=default \
  --country=br \
  --source /path/to/video.mp4
```

### Preview (motorcycle profile)
```bash
./build/src/alpr-tool preview \
  --profile=moto \
  --country=br \
  --source /path/to/video.mp4
```

### Preview (garage profile, more aggressive)
```bash
./build/src/alpr-tool preview \
  --profile=garagem \
  --country=br \
  --source /path/to/video.mp4
```

### Count only plates past the line (recommended)
OCR runs on every frame; **only plates whose position is past** the green line (vehicle already crossed) are counted and shown:

```bash
./build/src/alpr-tool preview \
  --country br \
  --source /path/to/video.mp4 \
  --plates-only-past-line 1 \
  --crossing-line-pct 50
```

- `--plates-only-past-line 1`: OCR always on; only plates with center **below** the line count.
- `--ocr-only-after-crossing 1`: disables OCR until the line is crossed; before that you’ll see “GATED (waiting crossing)”.
- `--crossing-line-pct 50`: horizontal line at 50% of frame height (default). Or set the line in pixels: `--line 0,Y,W,Y` (e.g. `--line 0,540,1920,540` for 1080p).
- `--crossing-fallback-sec 3`: if no crossing is detected after this many seconds, OCR is allowed anyway (avoids getting 0 plates when the tripwire never fires).

The **green line** drawn on the preview is the gate: OCR runs only after motion crosses it (or after the fallback time).

- **Estudo completo de configurações**: [docs/CONFIG_STUDY.md](docs/CONFIG_STUDY.md) — todas as chaves do .conf, CLI do `alpr-tool preview` e opções do core.

## GUI (Qt configurator)

Build:
```bash
cd build
cmake ..
make -j"$(nproc)" alpr-configurator
```

Run:
```bash
./build/src/tools/alpr-configurator/alpr-configurator
./build/src/tools/alpr-configurator/alpr-configurator --project artifacts/projects/demo.alprproj.json
```

Workflow (multi-source):
- File → New Project Wizard para criar um `.alprproj.json`
- Source dock: Add / Duplicate / Remove fontes (RTSP/vídeo/câmera)
- Cada source aponta para um `.conf` (ex.: `artifacts/configs/openalpr.garagem.conf`)
- Tabs de config (Runtime, Detection, OCR, ROI/Crossing, Prewarp, Logging, Advanced, Raw) preservam chaves desconhecidas (round-trip)
- Tools → Doctor valida runtime/cascade/tessdata e atualiza status
- File → Save/Save As grava o projeto e os `.conf` de cada fonte

---

## Performance recipes (practical and measurable)

### Recipe A — Use profiles (cheapest win)
Run the same input with different profiles and compare:
- `votes_emitted`
- `final_plate_count`
- wall-clock time of the run
- your existing `[report]` fields (fps, plates_found, plates_none, etc.)

### Recipe B — ROI/cropping in your app (biggest speed lever)
Even without any new detectors, **cropping** is the main way to reduce OCR cost:
- crop to the lane/line region before calling OCR
- keep pixel density high enough for plates (especially motorcycles)

### Recipe C — External detector (YOLO) + skip_detection (best control)
If your application uses YOLO/tracking:
- detect plate bbox externally
- pass bbox/crops into ALPR (OCR only)
- enable motorcycle/garage behavior using `profile` and temporal voting

This gives you:
- deterministic “only OCR when needed”
- stable output via voting
- minimal work inside the ALPR library

---

## Configuration notes

### skip_detection (optional)
`skip_detection` is off by default. Enable only when your application provides bboxes reliably:

```ini
skip_detection = 1
```

Use-case:
- YOLO detects the plate bbox
- you crop to that bbox and run OCR pipeline only

---

## Repository hot paths (most relevant files)

- Tool:
  - `src/tools/alpr_tool.cpp` — preview, profiles, voting, report fields

- OCR:
  - `src/openalpr/ocr/tesseract_ocr.cpp`
  - `src/openalpr/ocr/tesseract_ocr.h`
  - `src/openalpr/ocr/ocrfactory.cpp`
  - `src/openalpr/ocr/ocr.h`

- Build:
  - `src/cmake_modules/FindTesseract.cmake`
  - `src/openalpr/CMakeLists.txt`
  - `src/CMakeLists.txt`

---

## Roadmap (near-term, aligned to current direction)
- Move profile behavior deeper into the core (true multi-pass OCR per profile)
- Provide a first-class API for bbox input (OCR-only) for Python/Java wrappers
- Add automated “proof” command producing artifacts that demonstrate:
  - runtime assets ok
  - gating/voting behavior
  - measurable performance diffs by profile

---

## Disclaimer
This project is open source and not officially affiliated with OpenALPR Inc.

---

## 🇧🇷 Português

## O que existe hoje (recente / 2025)

### 1) Tag de profile (NOVO)
O `alpr-tool preview` suporta:

- `--profile=default|moto|garagem`

Profiles controlam a estratégia de OCR:

| Profile | Cenário | Comportamento atual |
|---|---|---|
| `default` | carro / geral | `ocr_burst_frames = 1` |
| `moto` | moto / placa pequena | `ocr_burst_frames = 6` + voto temporal |
| `garagem` | garagem / baixa velocidade | `ocr_burst_frames = 10` + voto temporal |

✅ Implementado em: `src/tools/alpr_tool.cpp`  
✅ Commit: `feat(tool): add profile tag with burst OCR and temporal voting`

---

### 2) Voto temporal (NOVO)
Para `moto` e `garagem`, o tool faz burst OCR e aplica majority vote.

Log:
```
[vote] profile=moto plate=ABC1D23 window=N
```

---

### 3) Métricas/Relatório (NOVO)
O report/JSON inclui:
- `profile`
- `ocr_burst_frames`
- `votes_emitted`
- `final_plate_count`

Isso permite comparar perfis por números.

---

### 4) Core com dependências mínimas (decisão fechada)
- Detector clássico OpenALPR
- OCR via Tesseract no core (C++)
- `skip_detection` existe (desligado por padrão) para cenários com bbox externa

---

### 5) Wrapper Java (coordenadas/bbox)
Melhorias recentes:
- API mais ergonômica para bounding box
- “bbox proof” em testes

---

## Como usar (CLI)

### Build
```bash
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
```

### Preview (mais rápido)
```bash
./build/src/alpr-tool preview --profile=default --country=br --source /path/to/video.mp4
```

### Preview (moto)
```bash
./build/src/alpr-tool preview --profile=moto --country=br --source /path/to/video.mp4
```

### Preview (garagem)
```bash
./build/src/alpr-tool preview --profile=garagem --country=br --source /path/to/video.mp4
```

### Contar só placas que já atravessaram a linha (recomendado)
OCR roda em todo frame; **só contam e aparecem** as placas cuja posição está **além** da linha verde (veículo já atravessou):

```bash
./build/src/alpr-tool preview --country br --source /path/to/video.mp4 \
  --plates-only-past-line 1 --crossing-line-pct 50
```

- `--plates-only-past-line 1`: OCR sempre ligado; só placas com centro **abaixo** da linha entram na contagem.
- `--ocr-only-after-crossing 1`: OCR só após cruzar; antes aparece “GATED (waiting crossing)”.
- `--crossing-line-pct 50`: linha horizontal a 50% da altura do frame. Ou use `--line 0,Y,W,Y` em pixels.
- `--crossing-fallback-sec 3`: se nenhum cruzamento for detectado após esses segundos, o OCR é liberado mesmo assim (evita ficar com 0 placas quando a tripwire nunca dispara).

A **linha verde** no preview é a porteira: o OCR só roda depois que o movimento cruza essa linha (ou após o tempo de fallback).

---

## Receitas de velocidade (sem achismo)

### Receita A — Comparar perfis
Use o mesmo vídeo e compare `default` vs `moto` vs `garagem` usando:
- `votes_emitted`
- `final_plate_count`
- tempo total de execução
- FPS e contadores do `[report]`

### Receita B — Crop/ROI na aplicação (maior ganho)
O maior ganho vem de **reduzir área** antes do OCR:
- recortar a região da pista/linha
- manter densidade de pixels (principalmente para moto)

### Receita C — YOLO fora + skip_detection (melhor controle)
Se você já usa YOLO/tracker:
- detecte bbox externamente
- use `skip_detection` e rode só OCR
- use `profile=moto` / `garagem` para burst+voto quando necessário

---

## skip_detection (opcional)
```ini
skip_detection = 1
```
Use apenas quando sua aplicação fornece bboxes confiáveis.

---

## Arquivos principais
- `src/tools/alpr_tool.cpp` — profiles, voto, métricas
- `src/openalpr/ocr/*` — Tesseract OCR core
- `src/cmake_modules/FindTesseract.cmake` — find/link Tesseract/Leptonica

---

## Aviso Legal
Projeto open source, sem afiliação oficial com OpenALPR Inc.
