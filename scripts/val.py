#!/usr/bin/env python3
"""Validate a trained checkpoint with the manuscript protocol."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT

from pdfn.protocol import val_args



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(ROOT / "configs/datasets/VisDrone.yaml"))
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=None)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    from pdfn.patch import register_pdfn_modules
    register_pdfn_modules()
    from ultralytics import YOLO
    kwargs = val_args(data=a.data, device=a.device, project=a.project, name=a.name)
    print(YOLO(a.weights).val(**kwargs))


if __name__ == "__main__":
    main()
