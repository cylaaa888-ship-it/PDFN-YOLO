from pathlib import Path

from .model import build_pdfn

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    model = build_pdfn(root / "configs" / "yolo11s-pdfn-v2.yaml", pretrained=None)
    model.model.info(detailed=True, verbose=True, imgsz=640)
