# PDFN-YOLO11s: source release for UAV small-object detection

This repository provides the paper-aligned implementation and reproducibility utilities for
**A Full-Link Progressive Dynamic Enhancement Network for UAV Small Object Detection**.
It targets `ultralytics==8.3.59` and keeps the model, ablations, training protocol, validation,
APsmall evaluation, multi-seed analysis, and release checks in one self-contained project.

Repository: `https://github.com/cylaaa888-ship-it/pdfn`

## What is included

- `pdfn/modules.py` — RFM-C3k2, MSPC, DFRM, SACM, and SAD-Head.
- `pdfn/patch.py` — Ultralytics 8.3.59 parser registration without editing site-packages.
- `pdfn/protocol.py` — one canonical copy of the manuscript training/validation settings.
- `configs/yolo11s-pdfn-v2.yaml` — complete PDFN topology.
- `configs/ablations/` — all eleven **Table 4** ablations (a)-(k).
- `scripts/train*.py` — main, ablation, cross-dataset, and multi-seed training entry points.
- `scripts/val.py`, `eval_coco_apsmall.py`, `benchmark_fps.py` — evaluation utilities.
- `scripts/analyze_losses.py` — final-N-epoch validation-loss analysis.
- `scripts/aggregate_multiseed.py` — mean ± sample-SD aggregation for repeated runs.
- `scripts/collect_environment.py` — environment/hardware metadata capture.
- `scripts/run_smoke_check.py` — model-build and synthetic-forward check.
- `tools/` — VisDrone conversion, label checks, YAML integrity and release checksum checks.
- `results_reference/` — manuscript-reported and revised measured summary values only.
- `PAPER_TO_CODE.md` — manuscript-to-code traceability map.
- `.github/workflows/static-checks.yml` — automatic syntax/YAML/static-test checks on GitHub.

## Important implementation/reproducibility notes

1. The integration is pinned to `ultralytics==8.3.59`. The custom parser mirrors the model
   parser interface of that version and refuses unsupported versions by default.
2. The eleven ablation YAMLs contain **no `nn.Identity` placeholders**. Each file expresses
   the minimal actual topology for that ablation. This keeps the YOLO11s baseline canonical
   and avoids artificial layer-index shifts that would reduce compatible pretrained-weight
   transfer from `yolo11s.pt`.
3. Cross-dataset training is dataset-specific retraining. Ultralytics rebuilds the detection
   model with `nc` from the selected dataset YAML during training; it is not zero-shot transfer.
4. The manuscript records `auto_augment=autoaugment`. It is retained in the argument set for
   protocol parity. In Ultralytics 8.3.59, detection uses the `v8_transforms` pipeline and
   `auto_augment` belongs to the classification augmentation path, so this argument does not
   add an AutoAugment transform to detection training.
5. Summary CSVs are not substitutes for raw evidence. This release does not fabricate
   checkpoints or per-epoch logs. If the journal requests them, deposit the retained real run
   directories/checkpoints separately.

## Environment

Paper environment: Python 3.8, PyTorch 2.0.0, CUDA 11.8, Ultralytics 8.3.59.

```bash
conda env create -f environment.yml
conda activate pdfn
pip install -e .
```

Alternative:

```bash
conda create -n pdfn python=3.8 -y
conda activate pdfn
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

Capture the actual reproduction environment before training:

```bash
python scripts/collect_environment.py --output runs/environment_report.json
```

## Prepare VisDrone2019-DET

```bash
python tools/prepare_visdrone.py \
  --source /data/VisDrone2019-DET \
  --output /data/VisDrone-YOLO \
  --copy-images
```

Set `path:` in `configs/datasets/VisDrone.yaml`, then check labels:

```bash
python tools/check_labels.py /data/VisDrone-YOLO/labels --classes 10
```

## Pre-flight verification

Dependency-light checks:

```bash
python -m compileall -q .
python tools/validate_model_configs.py
pytest -q
```

After installing the pinned Ultralytics environment, run the runtime checks:

```bash
python scripts/model_info.py
python scripts/run_smoke_check.py
python scripts/run_smoke_check.py --all-ablations
pytest -q
```

## Main training

```bash
python scripts/train.py \
  --data configs/datasets/VisDrone.yaml \
  --weights yolo11s.pt \
  --device 0
```

Defaults reproduce the manuscript protocol: 640×640, 300 epochs, batch 8, AdamW,
`lr0=0.001`, `patience=150`, `close_mosaic=10`, `multi_scale=False`, MixUp 0.05,
Copy-Paste 0, AMP off, deterministic execution, seed 42, and 2 workers.

Resume a real run:

```bash
python scripts/train.py --resume runs/pdfn/visdrone_pdfn_seed42/weights/last.pt --device 0
```

## Validation and APsmall

```bash
python scripts/val.py \
  --weights runs/pdfn/visdrone_pdfn_seed42/weights/best.pt \
  --data configs/datasets/VisDrone.yaml \
  --device 0
```

The validation protocol is `split=val`, 640 input, batch 1, confidence 0.001, IoU 0.7,
and `max_det=300`.

Convert official VisDrone validation annotations to COCO and evaluate APsmall:

```bash
python tools/visdrone_to_coco.py \
  --split-dir /data/VisDrone2019-DET-val \
  --output /data/VisDrone2019-DET-val/coco_gt.json

python scripts/eval_coco_apsmall.py \
  --ground-truth /data/VisDrone2019-DET-val/coco_gt.json \
  --predictions runs/detect/val/predictions.json \
  --output runs/detect/val/coco_metrics.json
```

## Table 4 ablations

One configuration:

```bash
python scripts/train_ablation.py configs/ablations/g_neck.yaml \
  --data configs/datasets/VisDrone.yaml --device 0
```

All eleven configurations:

```bash
python scripts/reproduce_all_ablations.py \
  --data configs/datasets/VisDrone.yaml --device 0
```

## Multi-seed stability experiment

Run YOLO11s and PDFN with seeds 0, 42, and 123:

```bash
python scripts/train_multiseed.py \
  --data configs/datasets/VisDrone.yaml \
  --weights yolo11s.pt \
  --device 0
```

After validating each retained `best.pt` and recording mAP50, mAP50-95 and APsmall in a
CSV with columns `model,seed,map50,map50_95,apsmall`, aggregate using sample SD (`n-1`):

```bash
python scripts/aggregate_multiseed.py \
  --input results_reference/multiseed_measured.csv \
  --output runs/multiseed_summary.csv
```

## Loss analysis

Compute the final-20-epoch mean ± sample SD directly from retained Ultralytics `results.csv`:

```bash
python scripts/analyze_losses.py \
  --input YOLO11s=runs/yolo11s/results.csv \
  --input PDFN=runs/pdfn/results.csv \
  --last 20 \
  --output runs/loss_summary.csv
```

## Cross-dataset retraining

Each dataset must be converted/configured separately and retrained rather than evaluated
zero-shot. Example:

```bash
python scripts/train_cross_dataset.py \
  --data configs/datasets/UAVDT.yaml \
  --weights yolo11s.pt \
  --device 0 \
  --name uavdt_pdfn_seed42
```

## Inference, export, efficiency

```bash
python scripts/predict.py --weights best.pt --source /path/to/images --device 0
python scripts/export.py --weights best.pt --format onnx
python scripts/benchmark_fps.py --weights best.pt --device cuda:0
```

FPS is batch-1 model forward only and excludes preprocessing and NMS, matching the manuscript.

## Revised measured stability summary

| Model | mAP@0.5 | mAP@0.5:0.95 | APsmall |
|---|---:|---:|---:|
| YOLO11s (3 runs) | 39.43 ± 0.15 | 23.57 ± 0.12 | 23.77 ± 0.15 |
| PDFN (3 runs) | 42.63 ± 0.15 | 25.80 ± 0.10 | 26.23 ± 0.15 |

Single seed-42 headline values remain YOLO11s 39.6/23.7/23.9 and PDFN 42.8/25.9/26.4.

## Release integrity

`SHA256SUMS.txt` covers the public files in this source release. Verify after download:

```bash
python tools/verify_release.py
```

See `VALIDATION.md` for what was and was not executable in the audit environment.
