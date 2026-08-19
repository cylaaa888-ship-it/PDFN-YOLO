#!/usr/bin/env python3
"""Train PDFN with manuscript defaults; explicit CLI overrides are logged by Ultralytics."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT

from pdfn.model import build_pdfn
from pdfn.protocol import train_args



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "configs/yolo11s-pdfn-v2.yaml"))
    p.add_argument("--data", default=str(ROOT / "configs/datasets/VisDrone.yaml"))
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/pdfn")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--resume", default=None, metavar="CHECKPOINT")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    name = a.name or f"visdrone_pdfn_seed{a.seed}"
    if a.resume:
        from pdfn.patch import register_pdfn_modules
        register_pdfn_modules()
        from ultralytics import YOLO
        model = YOLO(a.resume)
    else:
        model = build_pdfn(a.model, pretrained=a.weights)
    kwargs = train_args(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers,
        seed=a.seed, device=a.device, project=a.project, name=name,
        resume=bool(a.resume), plots=True, save=True, verbose=True,
    )
    model.train(**kwargs)


if __name__ == "__main__":
    main()
