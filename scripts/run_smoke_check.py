#!/usr/bin/env python3
"""Build model configs and run a synthetic forward pass without a dataset."""

from __future__ import annotations

import argparse
import torch
from _bootstrap import ROOT

from pdfn.model import build_pdfn



def main():
    p=argparse.ArgumentParser(); p.add_argument("--all-ablations",action="store_true"); p.add_argument("--imgsz",type=int,default=640); a=p.parse_args()
    configs=sorted((ROOT/"configs/ablations").glob("*.yaml")) if a.all_ablations else [ROOT/"configs/yolo11s-pdfn-v2.yaml"]
    for cfg in configs:
        print(f"building {cfg.name} ...")
        wrapper=build_pdfn(cfg,pretrained=None)
        wrapper.model.train()
        with torch.inference_mode():
            out=wrapper.model(torch.zeros(1,3,a.imgsz,a.imgsz))
        if not isinstance(out,list) or len(out)!=3:
            raise RuntimeError(f"unexpected output from {cfg.name}: {type(out)}")
        print(cfg.name, [tuple(x.shape) for x in out])


if __name__ == "__main__":
    main()
