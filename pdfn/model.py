"""Convenience constructors for PDFN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def build_pdfn(
    model_yaml: PathLike,
    pretrained: Optional[PathLike] = "yolo11s.pt",
    strict_version: bool = True,
):
    """Build PDFN and optionally transfer compatible YOLO11s pretrained weights."""
    from .patch import register_pdfn_modules

    register_pdfn_modules(strict_version=strict_version)
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    if pretrained:
        model.load(str(pretrained))
    return model
