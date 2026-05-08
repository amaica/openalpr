#!/usr/bin/env python3
"""
Recorta a placa (1º resultado) com base nas coordenadas do `alpr -j` na imagem completa,
compara tempo/confiança com uma nova passagem `alpr -j` só no recorte (detector ainda activo).

Nota: com a config BR/híbrido actual, `alpr --skip-detection` nestes PNGs costuma devolver
`results: []` (OCR não arranca no modo “só crop”); por isso o teste útil aqui é “full vs crop+detector”.

Uso (na raiz do repo):
  python3 scripts/crop_plate_from_alpr_json.py 1.png 2.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def alpr_json(repo: Path, extra_args: list[str]) -> dict:
    alpr = repo / "build/src/alpr"
    cfg = repo / "config/openalpr.conf.defaults"
    p = subprocess.run(
        [str(alpr), "-c", "br", "--config", str(cfg), *extra_args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"alpr failed rc={p.returncode} stderr={p.stderr!r}")
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            return json.loads(ln)
    raise RuntimeError("no JSON line in alpr stdout")


def bbox_from_coords(coords: list, pad_frac: float = 0.35, pad_min: int = 14) -> tuple[int, int, int, int]:
    xs = [c["x"] for c in coords]
    ys = [c["y"] for c in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    pad = max(pad_min, int(max(w, h) * pad_frac))
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def crop_save(src: Path, out: Path, box: tuple[int, int, int, int]) -> tuple[int, int]:
    im = Image.open(src).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid crop after clamp: {(x0, y0, x1, y1)} img={W}x{H}")
    im.crop((x0, y0, x1 + 1, y1 + 1)).save(out)
    return x1 - x0 + 1, y1 - y0 + 1


def summarize(label: str, data: dict) -> None:
    r = (data.get("results") or [{}])[0] if data.get("results") else {}
    plate = r.get("plate")
    conf = r.get("confidence")
    det_ms = r.get("processing_time_ms")
    tot = data.get("processing_time_ms")
    iw, ih = data.get("img_width"), data.get("img_height")
    print(f"  {label}: plate={plate!r} conf={conf} det_ms={det_ms} total_ms={tot} img={iw}x{ih}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--try-skip-detection",
        action="store_true",
        help="Also run alpr --skip-detection on the crop (often empty with BR hybrid).",
    )
    args = ap.parse_args()

    repo = repo_root()
    out_dir = args.out_dir or (repo / "artifacts/plate_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel in args.images:
        src = rel if rel.is_absolute() else (repo / rel)
        if not src.is_file():
            print(f"SKIP missing: {src}", file=sys.stderr)
            continue

        full = alpr_json(repo, ["-j", str(src)])
        results = full.get("results") or []
        if not results:
            print(f"NO_PLATE on full image: {src}", file=sys.stderr)
            continue
        r0 = results[0]
        coords = r0.get("coordinates") or []
        if len(coords) < 4:
            print(f"bad coordinates: {src}", file=sys.stderr)
            continue

        stem = src.stem
        crop_path = out_dir / f"{stem}_plate_crop.png"
        box = bbox_from_coords(coords)
        w, h = crop_save(src, crop_path, box)

        crop_data = alpr_json(repo, ["-j", str(crop_path)])

        print(f"\n=== {src.name} ===")
        summarize("full_image ", full)
        summarize(f"crop_only  ({w}x{h} -> {crop_path.name})", crop_data)

        if args.try_skip_detection:
            try:
                skip = alpr_json(repo, ["--skip-detection", "-j", str(crop_path)])
            except Exception as e:
                print(f"  skip_detection: error {e}")
            else:
                n = len(skip.get("results") or [])
                print(f"  skip_detection: {n} result(s) total_ms={skip.get('processing_time_ms')}")
                if n:
                    summarize("  skip_first", skip)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
