#!/usr/bin/env python3
"""Retrain PDFN independently on UAVDT, AI-TOD, or TinyPerson."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT

from pdfn.model import build_pdfn
from pdfn.protocol import train_args



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset YAML")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/pdfn_cross_dataset")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    model = build_pdfn(ROOT / "configs/yolo11s-pdfn-v2.yaml", pretrained=a.weights)
    model.train(**train_args(
        data=a.data, seed=a.seed, epochs=a.epochs, batch=a.batch, workers=a.workers,
        device=a.device, project=a.project, name=a.name, plots=True, save=True, verbose=True,
    ))


if __name__ == "__main__":
    main()
