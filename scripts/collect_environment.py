#!/usr/bin/env python3
"""Record software/hardware metadata for a reproducible training run."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path


def version(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return None


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=Path("environment_report.json")); a=p.parse_args()
    report={
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {x:version(x) for x in ("torch","torchvision","ultralytics","numpy","opencv-python","Pillow","PyYAML","pycocotools")},
    }
    try:
        import torch
        report["torch_cuda_version"]=torch.version.cuda
        report["cuda_available"]=torch.cuda.is_available()
        report["cuda_device_count"]=torch.cuda.device_count()
        report["cuda_devices"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        report["cudnn_version"]=torch.backends.cudnn.version()
    except Exception as exc:
        report["torch_probe_error"]=repr(exc)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(a.output.resolve())


if __name__ == "__main__":
    main()
