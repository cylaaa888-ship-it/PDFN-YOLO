#!/usr/bin/env bash
set -euo pipefail
python scripts/train.py --data configs/datasets/VisDrone.yaml --weights yolo11s.pt --device 0
