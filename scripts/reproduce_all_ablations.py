#!/usr/bin/env python3
"""Sequentially train all eleven Table-4 ablation configurations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "configs/datasets/VisDrone.yaml"))
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/pdfn_ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--start", default="a_baseline.yaml")
    a = p.parse_args()

    configs = sorted((ROOT / "configs/ablations").glob("*.yaml"))
    names = [p.name for p in configs]
    if a.start not in names:
        raise ValueError(f"--start must be one of: {', '.join(names)}")
    configs = configs[names.index(a.start):]
    for config in configs:
        cmd = [sys.executable, str(ROOT/"scripts/train_ablation.py"), str(config),
               "--data", a.data, "--weights", a.weights, "--device", a.device,
               "--project", a.project, "--seed", str(a.seed), "--epochs", str(a.epochs),
               "--batch", str(a.batch), "--workers", str(a.workers)]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
