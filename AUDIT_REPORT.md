# Code audit and enhancement report

## Material issue fixed

The original ablation YAMLs inserted `nn.Identity` layers where DFRM/SACM were disabled so that
all variants shared layer numbers. Although these layers are mathematically no-ops, they shift
subsequent state-dict indices. Because experiments initialize from `yolo11s.pt`, this can reduce
matching of otherwise unchanged pretrained neck/head parameters and weakens initialization
fairness. All eleven ablations were regenerated as minimal real topologies with corrected layer
references. The baseline now follows the canonical YOLO11 head indexing.

## Reproducibility improvements

- Centralized paper training/validation defaults in `pdfn/protocol.py`.
- Added seed/epoch/batch/worker CLI controls while preserving paper defaults.
- Added reproducible three-seed runner and sample-SD aggregation.
- Added final-N-epoch validation-loss analysis from real Ultralytics `results.csv`.
- Added environment/hardware capture.
- Added YAML static validation, release checksum verification and runtime smoke-check entry point.
- Added robust pytest configuration so source-tree testing works before editable installation.
- Corrected manuscript mapping from “Table 3 ablations” to **Table 4 ablations**, and protocol
  mapping to **Table 2**.
- Added result-reference provenance notes, `.gitignore`, and `CITATION.cff`.

## Protocol note

`auto_augment="autoaugment"` remains in the canonical argument set because it appears in the
manuscript settings. In Ultralytics 8.3.59 the object-detection augmentation function
`v8_transforms` does not consume `auto_augment`; that option is used by the classification
augmentation path. Therefore its presence does not add AutoAugment to the detection pipeline.

## Remaining evidence boundary

No synthetic checkpoints, logs, or training curves were created. The enhanced source release can
reproduce and aggregate the experiments, but raw run evidence must come from the actual retained
training runs.
