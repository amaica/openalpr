# 🚘 OpenALPR 2025 — Brazil & Mercosur (Cars + Motorcycles, YOLO-powered)

> 🇺🇸 English version  
> 🇧🇷 Versão em Português logo abaixo

---

## 📌 Overview

This repository is a **deep architectural evolution** of the classic OpenALPR engine, bringing it to **2025 standards**, with a strong focus on:

- 🇧🇷 **Brazil & Mercosur plates**
- 🏍️ **Motorcycle plates (real detection + OCR)**
- ⚡ **High performance and real scalability**
- 🧠 **Modern AI integration (YOLOv8)**
- 🔌 **Plugable, configurable, production-ready architecture**

> **This is not just a fork.**  
> It is an incremental reengineering effort that preserves what works and replaces what aged.

---

## ✨ Key Features

### ✅ Native Brazil & Mercosur support
- Old Brazilian plates (LLLNNNN)
- Mercosur plates (LLLNLNN)
- Native `br2 → br` fallback
- No dependency on `eu/ad`
- Explicit, configurable and logged fallback logic

---

### 🏍️ Real motorcycle plate support
- YOLOv8-based detection
- Dedicated OCR profiles:
  - `br_moto.conf`
  - `br2_moto.conf`
- Automatic profile selection:
  - YOLO class (`plate_car` / `plate_moto`)
  - or bounding box aspect ratio
- Same 7-character validation rules

---

### 🧠 Modern YOLO-powered detection (plugable)
- YOLOv8 ONNX as primary detector
- Model loaded via config (no recompilation)
- Automatic hardware detection:
  - CPU
  - GPU (CUDA when available)
- Safe fallback to classic detector

---

### ⚡ Production-grade performance
- Process-based parallelism
- One YOLO + one ALPR instance per worker
- Linear scaling with CPU/GPU
- No shared state, no race conditions

---

### 🔌 2025-ready architecture
- Detector abstraction (classic / YOLO / future)
- Vehicle-type OCR profiles (car / motorcycle)
- Prepared for:
  - Modern OCR (ONNX)
  - Video tracking
  - Microservices
  - MCP / AI agents

---

## 🧩 Architecture (High Level)

