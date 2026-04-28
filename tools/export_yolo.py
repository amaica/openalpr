#!/usr/bin/env python3
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt to ONNX for OpenALPR fork")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Install with: pip install -U ultralytics", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Exporting to ONNX: {args.out} (imgsz={args.imgsz})")
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, dynamic=False, simplify=False)
    if exported is None:
        print("Export failed", file=sys.stderr)
        sys.exit(1)

    # ultralytics returns output path; move if needed
    if os.path.abspath(exported) != os.path.abspath(args.out):
        try:
          os.replace(exported, args.out)
        except OSError:
          import shutil
          shutil.copyfile(exported, args.out)
    print(f"ONNX saved to: {args.out}")
    sys.exit(0)

if __name__ == "__main__":
    main()

