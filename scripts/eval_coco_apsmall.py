#!/usr/bin/env python3
"""Evaluate COCO AP, including APsmall, from Ultralytics prediction JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    gt = COCO(str(args.ground_truth))
    dt = gt.loadRes(str(args.predictions))
    evaluator = COCOeval(gt, dt, iouType="bbox")
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    result = {
        "AP_50_95": float(evaluator.stats[0]),
        "AP_50": float(evaluator.stats[1]),
        "AP_75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR_max1": float(evaluator.stats[6]),
        "AR_max10": float(evaluator.stats[7]),
        "AR_max300": float(evaluator.stats[8]),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
