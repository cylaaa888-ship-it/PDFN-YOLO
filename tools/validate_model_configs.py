#!/usr/bin/env python3
"""Static validation for all PDFN YAML layer references and ablation flags."""
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def check(path):
    cfg=yaml.safe_load(path.read_text(encoding="utf-8")); layers=cfg["backbone"]+cfg["head"]
    for i,(source,_,_,_) in enumerate(layers):
        for ref in source if isinstance(source,list) else [source]:
            if ref==-1: continue
            if ref>=0 and ref>=i: raise ValueError(f"{path.name}: layer {i} -> {ref}")
            if ref<0 and i+ref<0: raise ValueError(f"{path.name}: layer {i} -> {ref}")
    if any(x[2]=="nn.Identity" for x in layers): raise ValueError(f"{path.name}: nn.Identity placeholder found")
    return len(layers)

def main():
    paths=[ROOT/"configs/yolo11s-pdfn-v2.yaml",*sorted((ROOT/"configs/ablations").glob("*.yaml"))]
    for p in paths: print(f"OK {p.relative_to(ROOT)} layers={check(p)}")
if __name__=="__main__": main()
