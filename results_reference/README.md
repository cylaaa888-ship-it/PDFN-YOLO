# Result-reference files

This directory separates **reported/measured summary values** from executable code.

- `paper_reported_results.csv`: controlled YOLO11s/PDFN headline values in the manuscript.
- `paper_reported_comparison.csv`: manuscript comparison table; literature rows are contextual, not hardware-normalized reruns.
- `paper_reported_ablation.csv`: Table 4 ablation summary.
- `paper_reported_cross_dataset.csv`: dataset-specific retraining summary.
- `multiseed_measured.csv`: three-run stability measurements (seeds 0, 42, 123) confirmed for the revised study.
- `loss_measured_summary.csv`: final-20-epoch validation-loss summary confirmed for the revised study.

Raw checkpoints and per-epoch training logs are **not synthesized** in this release. Use
`scripts/train_multiseed.py`, `scripts/aggregate_multiseed.py`, and `scripts/analyze_losses.py`
to regenerate/aggregate those artifacts in the pinned environment. If the journal requests raw
logs, upload the retained real run directories separately rather than constructing replacement logs.
