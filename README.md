# 🚘 OpenALPR 2025 — Brazil & Mercosur (YOLOv8)

Modernized OpenALPR engine with **native Brazil/Mercosur support**, **motorcycle plates**, and **YOLOv8-based detection**, built for **production in 2025**.

> 🇺🇸 English  
> 🇧🇷 Português abaixo

---

## English

### What is this?
A **production-grade evolution** of the classic OpenALPR engine, focused on:
- Brazil & Mercosur plates
- Cars and motorcycles
- High performance C++
- Modern AI detection (YOLOv8)
- Clean, extensible architecture (no hacks)

### Key features
- Native Brazil support (no `eu/ad` dependency)
- Mercosur plates: `br2 → br` fallback
- Motorcycle plate support with dedicated OCR profiles
- YOLOv8 ONNX detector (plugable by config)
- Automatic CPU/GPU backend selection
- Process-based parallelism (safe & scalable)

### Build
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
