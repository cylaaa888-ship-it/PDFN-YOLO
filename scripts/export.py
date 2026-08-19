#!/usr/bin/env python3
"""Export a trained PDFN checkpoint."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  # adds repository root to sys.path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=("onnx", "engine", "torchscript", "openvino"))
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    from pdfn.patch import register_pdfn_modules
    register_pdfn_modules()
    from ultralytics import YOLO

    YOLO(args.weights).export(
        format=args.format,
        imgsz=640,
        batch=1,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
    )


if __name__ == "__main__":
    main()
