"""PDFN: a paper-grounded YOLO11s implementation for UAV small-object detection."""

__version__ = "0.2.0"


def register_pdfn_modules(strict_version: bool = True) -> None:
    """Lazily register PDFN modules into the pinned Ultralytics parser."""
    from .patch import register_pdfn_modules as _register

    _register(strict_version=strict_version)


__all__ = ("register_pdfn_modules",)
