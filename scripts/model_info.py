#!/usr/bin/env python3
"""Print parameters/GFLOPs and compare with the paper's reference numbers."""


from _bootstrap import ROOT

from pdfn.model import build_pdfn


if __name__ == "__main__":
    model = build_pdfn(ROOT / "configs/yolo11s-pdfn-v2.yaml", pretrained=None)
    model.model.info(detailed=True, verbose=True, imgsz=640)
    print("Paper reference: 8.9 M parameters, 21.8 GFLOPs, 123 FPS.")
    print("Re-measure GFLOPs/FPS in the pinned environment; small reporting differences can arise from library counting conventions.")
