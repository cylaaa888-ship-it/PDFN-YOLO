#!/usr/bin/env python3
"""Verify SHA256SUMS.txt for the public source release."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]

def main():
    errors = []
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        path = ROOT / rel.strip()
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"mismatch: {rel}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("SHA256 verification passed")

if __name__ == "__main__":
    main()
