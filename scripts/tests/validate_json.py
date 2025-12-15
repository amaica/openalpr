#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"[validate_json][error] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        die("usage: validate_json.py <json_file>")
    path = Path(sys.argv[1])
    if not path.is_file():
        die(f"file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        die(f"invalid JSON: {exc}")
    if not data:
        die("JSON empty")
    # Light schema expectation
    if isinstance(data, dict):
        if "results" in data and not isinstance(data["results"], list):
            die("results field exists but is not a list")
    print("[validate_json] OK")


if __name__ == "__main__":
    main()

