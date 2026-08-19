# Validation status of the enhanced release

## Completed in the current code audit

1. Python syntax compilation for all `.py` files.
2. Static YAML parsing and forward-reference validation for the full model and all eleven
   ablation configurations.
3. Verification that no ablation YAML contains `nn.Identity` placeholder layers.
4. Verification that the baseline uses the canonical 24-layer YOLO11 topology and Detect inputs
   `[16, 19, 22]`, while the full model contains four RFM-C3k2 modules, one MSPC, four DFRMs,
   four SACMs and one SAD-Head.
5. Dependency-light pytest suite: configuration tests pass; runtime module tests are skipped when
   the pinned Ultralytics dependency is unavailable rather than failing at collection.
6. Multi-seed aggregation script reproduces the revised measured Mean ± sample-SD values.
7. Source-to-manuscript table-number mapping and release documentation were corrected.

## Not executable in the current audit sandbox

The sandbox does not contain `ultralytics==8.3.59`, `pycocotools`, CUDA, the datasets, or the
retained training checkpoints/logs. External package installation is unavailable in this sandbox.
Therefore the current audit cannot truthfully claim a new GPU training reproduction, GFLOPs/FPS
remeasurement, or a fresh runtime forward pass under the pinned environment.

## Required final pre-submission commands in the real pinned environment

```bash
conda env create -f environment.yml
conda activate pdfn
pip install -e .
python scripts/collect_environment.py --output runs/environment_report.json
pytest -q
python scripts/model_info.py
python scripts/run_smoke_check.py --all-ablations
```

Then retain the command output with the journal submission or repository release. For experiment
reproduction, keep the actual `results.csv`, `args.yaml`, `best.pt`, `last.pt`, validation JSON,
and environment report for every run used in a table.
