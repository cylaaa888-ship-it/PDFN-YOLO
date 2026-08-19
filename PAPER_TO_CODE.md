# Paper-to-code mapping

| Manuscript item | Code implementation |
|---|---|
| Fig. 1 full PDFN | `configs/yolo11s-pdfn-v2.yaml` |
| RFM-C3k2, Fig. 2, Eqs. (3)-(7) | `pdfn/modules.py::RFMBlock`, `RFMC3k2` |
| MSPC, Fig. 3, Eqs. (8)-(12) | `pdfn/modules.py::MSPC` |
| DFRM, Fig. 4, Eqs. (13)-(17) | `pdfn/modules.py::DFRM` |
| SACM, Fig. 5, Eqs. (18)-(21) | `pdfn/modules.py::SACM` |
| SAD-Head, Fig. 6, Eqs. (22)-(29) | `pdfn/modules.py::SADDetect` |
| Table 2 reproducibility settings | `pdfn/protocol.py`, `scripts/train.py`, `scripts/val.py` |
| Table 3 controlled comparison values | `results_reference/paper_reported_comparison.csv` |
| Table 4 ablations (a)-(k) | `configs/ablations/*.yaml`, `scripts/reproduce_all_ablations.py` |
| Table 5 partial/full-link paths | combinations represented by `d_backbone.yaml`, `g_neck.yaml`, `h_sad_head.yaml`, `i_backbone_neck.yaml`, `j_neck_sad_head.yaml`, `k_pdfn_full.yaml` |
| Table 6 cross-dataset retraining | `scripts/train_cross_dataset.py`, `configs/datasets/*.yaml` |
| APsmall | `tools/visdrone_to_coco.py`, `scripts/eval_coco_apsmall.py` |
| Forward-only FPS | `scripts/benchmark_fps.py` |
| Revised multi-seed experiment | `scripts/train_multiseed.py`, `scripts/aggregate_multiseed.py`, `results_reference/multiseed_measured.csv` |
| Revised validation-loss analysis | `scripts/analyze_losses.py`, `results_reference/loss_measured_summary.csv` |
| Environment/release traceability | `scripts/collect_environment.py`, `tools/verify_release.py`, `SHA256SUMS.txt` |

## Explicit implementation choices

The release makes the reconstruction choices stated in the revised manuscript explicit:

- MSPC compression ratio: `0.25`.
- RFM/DFRM gate reduction ratio: `4`.
- DFRM channel alignment: two-group 1×1 convolution when valid, with GCD fallback.
- DFRM bilinear resizing: `align_corners=False`.
- SACM statistical projection: per-channel grouped 1×1 convolution followed by BN.
- SAD-Head regression stem: depthwise-separable convolutions plus depthwise local compensation.
- Ultralytics integration target: `8.3.59`.

Ablation YAMLs intentionally use minimal real topologies rather than no-op identity placeholders.
This preserves a canonical YOLO11s baseline and avoids avoidable state-dict index shifts during
pretrained initialization.
