#!/usr/bin/env python3
import argparse
import os
import sys


def die(msg: str) -> None:
    print(f"[export_yolo][error] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt to ONNX")
    parser.add_argument("--pt", required=True, help="Path to YOLOv8 .pt model")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    args = parser.parse_args()

    if not os.path.isfile(args.pt):
        die(f"Model file not found: {args.pt}")

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # noqa: BLE001
        die(f"ultralytics not available: {exc}. Install with: pip install ultralytics")

    model = YOLO(args.pt)
    result = model.export(format="onnx", imgsz=args.imgsz, opset=12, dynamic=False, simplify=False, half=False)
    # Ultralytics returns path when successful
    onnx_path = os.path.abspath(args.out)
    if result and isinstance(result, str) and os.path.isfile(result):
        os.replace(result, onnx_path)
    elif os.path.isfile("model.onnx"):
        os.replace("model.onnx", onnx_path)
    else:
        die("Export did not produce an ONNX file")

    if not os.path.isfile(onnx_path) or os.path.getsize(onnx_path) == 0:
        die(f"ONNX output missing or empty: {onnx_path}")

    print(f"[export_yolo] ONNX generated at {onnx_path}")


if __name__ == "__main__":
    main()

