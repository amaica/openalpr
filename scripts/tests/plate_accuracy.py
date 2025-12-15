#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import List, Dict


def normalize_plate(p: str) -> str:
  return "".join(p.split()).upper()


def run_alpr(img: str, bin_path: str) -> Dict:
  try:
    proc = subprocess.run([bin_path, "-j", img], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout or "{}")
  except Exception as e:
    return {"error": str(e)}


def main():
  parser = argparse.ArgumentParser(description="Plate accuracy calculator using ground truth CSV.")
  parser.add_argument("--csv", required=True, help="ground truth CSV path (path,expected_plate)")
  parser.add_argument("--out", required=True, help="output JSON report path")
  parser.add_argument("--bin", default="./build/src/alpr", help="alpr binary with JSON output")
  args = parser.parse_args()

  if not os.path.isfile(args.csv):
    print(f"CSV not found: {args.csv}", file=sys.stderr)
    sys.exit(1)
  if not os.path.isfile(args.bin):
    print(f"Binary not found: {args.bin}", file=sys.stderr)
    sys.exit(1)

  samples: List[Dict] = []
  with open(args.csv, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
      if not row or row[0].startswith("#") or len(row) < 1:
        continue
      path = row[0].strip()
      expected = row[1].strip() if len(row) > 1 else ""
      samples.append({"path": path, "expected": expected})

  total = len(samples)
  labeled = 0
  correct = 0
  incorrect = 0
  no_plate = 0
  needs_label = 0
  cases = []

  for s in samples:
    res = run_alpr(s["path"], args.bin)
    top1 = ""
    if "results" in res and res["results"]:
      top1 = res["results"][0].get("plate", "")
    elif res.get("error"):
      top1 = ""
    if not s["expected"]:
      needs_label += 1
      status = "needs_label"
    else:
      labeled += 1
      if not top1:
        no_plate += 1
        status = "no_plate"
      elif normalize_plate(top1) == normalize_plate(s["expected"]):
        correct += 1
        status = "correct"
      else:
        incorrect += 1
        status = "incorrect"
    cases.append({
      "path": s["path"],
      "expected": s["expected"],
      "predicted": top1,
      "status": status,
    })

  accuracy = (correct / labeled) if labeled > 0 else 0.0
  report = {
    "total_samples": total,
    "labeled_samples": labeled,
    "needs_label": needs_label,
    "correct": correct,
    "incorrect": incorrect,
    "no_plate": no_plate,
    "accuracy_top1": accuracy,
    "cases": cases,
  }

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, "w") as f:
    json.dump(report, f, indent=2)

  txt_out = args.out.replace(".json", ".txt")
  with open(txt_out, "w") as f:
    f.write(f"total={total} labeled={labeled} needs_label={needs_label} correct={correct} incorrect={incorrect} no_plate={no_plate} accuracy_top1={accuracy:.3f}\n")
    worst = [c for c in cases if c["status"] == "incorrect"]
    f.write("worst_cases (pred!=expected):\n")
    for c in worst[:10]:
      f.write(f"{c['path']} expected={c['expected']} predicted={c['predicted']}\n")

  print(f"Wrote {args.out} and {txt_out}")


if __name__ == "__main__":
  main()

