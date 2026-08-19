#!/usr/bin/env python3
"""Convert official VisDrone2019-DET annotations to Ultralytics YOLO format.

Expected source layout examples:
  VisDrone2019-DET-train/images/*.jpg
  VisDrone2019-DET-train/annotations/*.txt
  VisDrone2019-DET-val/images/*.jpg
  VisDrone2019-DET-val/annotations/*.txt
  VisDrone2019-DET-test-dev/images/*.jpg

VisDrone category IDs 1..10 are mapped to YOLO IDs 0..9. Category 0
(ignored regions) and category 11 (others) are skipped.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image
from tqdm import tqdm

SPLIT_CANDIDATES = {
    "train": ("VisDrone2019-DET-train", "train"),
    "val": ("VisDrone2019-DET-val", "val"),
    "test-dev": ("VisDrone2019-DET-test-dev", "test-dev", "test"),
}


def find_split(root: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def parse_row(row: str) -> Optional[Tuple[float, float, float, float, int]]:
    fields = row.strip().rstrip(",").split(",")
    if len(fields) < 6:
        return None
    x, y, w, h = map(float, fields[:4])
    category = int(float(fields[5]))
    if category < 1 or category > 10 or w <= 0 or h <= 0:
        return None
    return x, y, w, h, category - 1


def convert_split(source: Path, output: Path, split_name: str, copy_images: bool) -> None:
    images_dir = source / "images"
    annotations_dir = source / "annotations"
    out_images = output / "images" / split_name
    out_labels = output / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    for image_path in tqdm(images, desc=split_name):
        destination = out_images / image_path.name
        if not destination.exists():
            if copy_images:
                shutil.copy2(image_path, destination)
            else:
                try:
                    destination.symlink_to(image_path.resolve())
                except OSError:
                    shutil.copy2(image_path, destination)

        label_path = out_labels / f"{image_path.stem}.txt"
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            label_path.write_text("", encoding="utf-8")
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        labels = []
        for row in annotation_path.read_text(encoding="utf-8-sig").splitlines():
            parsed = parse_row(row)
            if parsed is None:
                continue
            x, y, w, h, cls = parsed
            x1 = min(max(x, 0.0), float(width))
            y1 = min(max(y, 0.0), float(height))
            x2 = min(max(x + w, 0.0), float(width))
            y2 = min(max(y + h, 0.0), float(height))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            xc = (x1 + x2) / (2.0 * width)
            yc = (y1 + y2) / (2.0 * height)
            wn = w / width
            hn = h / height
            labels.append(f"{cls} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Directory containing official VisDrone split folders")
    parser.add_argument("--output", required=True, type=Path, help="YOLO-format output root")
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of symlink images")
    args = parser.parse_args()

    found = 0
    for output_name, candidates in SPLIT_CANDIDATES.items():
        split = find_split(args.source, candidates)
        if split is None:
            print(f"skip {output_name}: no matching source directory")
            continue
        convert_split(split, args.output, output_name, args.copy_images)
        found += 1
    if found == 0:
        raise FileNotFoundError("No recognized VisDrone2019-DET split directories were found.")
    print(f"Converted dataset written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
