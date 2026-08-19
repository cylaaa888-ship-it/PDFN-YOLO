#!/usr/bin/env python3
"""Run PDFN inference on an image, directory, video, or stream."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  # adds repository root to sys.path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    from pdfn.patch import register_pdfn_modules
    register_pdfn_modules()
    from ultralytics import YOLO

    YOLO(args.weights).predict(
        source=args.source,
        imgsz=640,
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
