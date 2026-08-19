"""Canonical training and validation settings reported in the revised manuscript."""

from __future__ import annotations

from typing import Any, Dict

PAPER_TRAIN_ARGS: Dict[str, Any] = {
    "imgsz": 640,
    "epochs": 300,
    "batch": 8,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "patience": 150,
    "close_mosaic": 10,
    "multi_scale": False,
    "augment": True,
    # Retained for argument parity with the manuscript. In Ultralytics 8.3.59,
    # auto_augment belongs to the classification augmentation path and is not
    # consumed by the detection v8_transforms pipeline.
    "auto_augment": "autoaugment",
    "mixup": 0.05,
    "copy_paste": 0.0,
    "amp": False,
    "seed": 42,
    "deterministic": True,
    "workers": 2,
}

PAPER_VAL_ARGS: Dict[str, Any] = {
    "split": "val",
    "imgsz": 640,
    "batch": 1,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "plots": True,
    "save_json": True,
}


def train_args(**overrides: Any) -> Dict[str, Any]:
    """Return a copy of the paper training settings with explicit overrides."""
    args = dict(PAPER_TRAIN_ARGS)
    args.update({k: v for k, v in overrides.items() if v is not None})
    return args


def val_args(**overrides: Any) -> Dict[str, Any]:
    """Return a copy of the paper validation settings with explicit overrides."""
    args = dict(PAPER_VAL_ARGS)
    args.update({k: v for k, v in overrides.items() if v is not None})
    return args
