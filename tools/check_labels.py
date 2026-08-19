#!/usr/bin/env python3
"""Basic integrity checks for YOLO label files."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--classes", type=int, default=10)
    args = parser.parse_args()

    files = list(args.labels.rglob("*.txt"))
    errors = []
    boxes = 0
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{path}:{line_no}: expected 5 fields")
                continue
            cls = int(parts[0])
            values = list(map(float, parts[1:]))
            if not 0 <= cls < args.classes:
                errors.append(f"{path}:{line_no}: invalid class {cls}")
            if any(value < 0 or value > 1 for value in values):
                errors.append(f"{path}:{line_no}: coordinates outside [0,1]")
            boxes += 1
    print(f"files={len(files)}, boxes={boxes}, errors={len(errors)}")
    for error in errors[:50]:
        print(error)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
