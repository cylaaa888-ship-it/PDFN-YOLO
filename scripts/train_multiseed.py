#!/usr/bin/env python3
"""Run the revised-paper 3-seed stability experiment for YOLO11s and PDFN."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "YOLO11s": ROOT / "configs/ablations/a_baseline.yaml",
    "PDFN": ROOT / "configs/yolo11s-pdfn-v2.yaml",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123])
    p.add_argument("--data", default=str(ROOT / "configs/datasets/VisDrone.yaml"))
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/pdfn_multiseed")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    for model_name in a.models:
        for seed in a.seeds:
            run_name = f"{model_name.lower()}_seed{seed}"
            cmd = [sys.executable, str(ROOT/"scripts/train_ablation.py"), str(CONFIGS[model_name]),
                   "--data", a.data, "--weights", a.weights, "--device", a.device,
                   "--project", a.project, "--name", run_name, "--seed", str(seed),
                   "--epochs", str(a.epochs), "--batch", str(a.batch), "--workers", str(a.workers)]
            print("RUN:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
