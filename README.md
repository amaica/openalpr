# 🚘 OpenALPR 2025 — Brazil & Mercosur  
### Cars + Motorcycles • YOLOv8 • Production-ready C++

> 🇺🇸 **English version**  
> 🇧🇷 **Versão em Português logo abaixo**

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Configuration](#configuration)
- [Build](#build)
- [Usage](#usage)
- [Project Status](#project-status)
- [Philosophy](#philosophy)
- [Disclaimer](#disclaimer)
- [Português](#português)

---

## Overview

This repository is a **deep architectural evolution** of the classic OpenALPR engine, bringing it to **2025 production standards**, with a strong focus on:

- 🇧🇷 **Brazil & Mercosur license plates**
- 🏍️ **Motorcycle plates (real detection and OCR)**
- ⚡ **High performance and real scalability**
- 🧠 **Modern AI-based detection (YOLOv8)**
- 🔌 **Plugable, configurable, and extensible architecture**

> **This is not just a fork.**  
> It is an incremental reengineering effort that preserves what works and replaces what aged — without shortcuts or fragile hacks.

---

## ✨ Key Features

### ✅ Native Brazil & Mercosur Support
- Old Brazilian plates: **LLLNNNN**
- Mercosur plates: **LLLNLNN**
- Native hybrid pipeline: **br2 → br**
- No dependency on `eu/ad`
- Explicit, deterministic, and logged fallback logic

---

### 🏍️ Real Motorcycle Plate Support
- YOLOv8-based detection
- Dedicated OCR profiles for motorcycles:
  - `br_moto.conf`
  - `br2_moto.conf`
- Automatic selection:
  - YOLO class (`plate_car` / `plate_moto`)
  - or bounding box aspect ratio
- Same 7-character validation rules
- No hacks or duplicated OCR logic

---

### 🧠 Modern YOLO-Powered Detection (Plugable)
- YOLOv8 ONNX as primary detector
- Models loaded by **config path** (no recompilation)
- Automatic hardware detection:
  - CPU
  - GPU (CUDA, when available)
- Safe fallback to classic detector
- Detector selection:
  ```ini
  detector_type = auto   # auto | yolo | classic
