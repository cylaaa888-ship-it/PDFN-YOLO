#!/usr/bin/env python3
"""Train one ablation configuration under the shared manuscript protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

from pdfn.model import build_pdfn
from pdfn.protocol import train_args



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", help="YAML under configs/ablations or an absolute path")
    p.add_argument("--data", default=str(ROOT / "configs/datasets/VisDrone.yaml"))
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/pdfn_ablation")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    config = Path(args.config)
    if not config.exists():
        config = ROOT / "configs" / "ablations" / args.config
    if not config.exists():
        raise FileNotFoundError(config)
    model = build_pdfn(config, pretrained=args.weights)
    name = args.name or f"{config.stem}_seed{args.seed}"
    model.train(**train_args(
        data=args.data, seed=args.seed, epochs=args.epochs, batch=args.batch,
        workers=args.workers, device=args.device, project=args.project, name=name,
        plots=True, save=True, verbose=True,
    ))


if __name__ == "__main__":
    main()
