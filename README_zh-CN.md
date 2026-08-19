# PDFN-YOLO11s 期刊源码与复现实验工程

本项目对应论文 **A Full-Link Progressive Dynamic Enhancement Network for UAV Small Object Detection**，
固定适配 `ultralytics==8.3.59`，用于期刊源码提交、模型复现、消融实验、多随机种子稳定性实验和
Loss 分析。

仓库地址：`https://github.com/cylaaa888-ship-it/pdfn`

## 本次完善后的主要内容

- `pdfn/modules.py`：RFM-C3k2、MSPC、DFRM、SACM、SAD-Head。
- `pdfn/patch.py`：适配 Ultralytics 8.3.59 的自定义解析器。
- `pdfn/protocol.py`：论文训练/验证参数统一定义，避免不同脚本参数漂移。
- `configs/yolo11s-pdfn-v2.yaml`：PDFN 完整结构。
- `configs/ablations/`：论文**表4**的 11 组消融配置。
- `scripts/train_multiseed.py`：seed=0、42、123 多随机种子训练。
- `scripts/aggregate_multiseed.py`：自动计算 Mean ± 样本标准差。
- `scripts/analyze_losses.py`：从真实 `results.csv` 统计最后 N 轮验证 Loss。
- `scripts/collect_environment.py`：记录 Python、PyTorch、CUDA、GPU 等复现信息。
- `scripts/run_smoke_check.py`：构建模型并进行随机张量前向检查。
- `tools/validate_model_configs.py`：无需 GPU/Ultralytics 的 YAML 静态检查。
- `tools/verify_release.py`：发布包 SHA-256 完整性校验。
- `results_reference/`：论文结果与本轮大修实测汇总值，不伪造训练日志或权重。
- `.github/workflows/static-checks.yml`：GitHub 自动语法、YAML 与静态测试。

## 关键修正

旧版消融 YAML 为保持统一层号，在关闭 DFRM/SACM 时加入了 `nn.Identity` 占位层。虽然前向计算
等价，但会改变后续 state-dict 层编号，从而降低 `yolo11s.pt` 对未修改 neck/head 层的预训练权重
匹配率。本版已将 11 组消融全部改为**无占位层的实际最小拓扑**：基线恢复为标准 YOLO11s 的
24 层索引结构，其他配置按真实增加模块重新计算引用层号，更符合论文“统一以 yolo11s.pt 初始化”
的公平性要求。

## 环境

论文环境：Python 3.8、PyTorch 2.0.0、CUDA 11.8、Ultralytics 8.3.59。

```bash
conda env create -f environment.yml
conda activate pdfn
pip install -e .
```

建议训练前保存环境信息：

```bash
python scripts/collect_environment.py --output runs/environment_report.json
```

## 数据集

```bash
python tools/prepare_visdrone.py \
  --source /data/VisDrone2019-DET \
  --output /data/VisDrone-YOLO \
  --copy-images
```

然后修改 `configs/datasets/VisDrone.yaml` 中的 `path`，并运行：

```bash
python tools/check_labels.py /data/VisDrone-YOLO/labels --classes 10
```

## 训练前检查

```bash
python -m compileall -q .
python tools/validate_model_configs.py
pytest -q
```

在固定环境安装完成后再运行：

```bash
python scripts/model_info.py
python scripts/run_smoke_check.py --all-ablations
pytest -q
```

## 主实验

```bash
python scripts/train.py \
  --data configs/datasets/VisDrone.yaml \
  --weights yolo11s.pt \
  --device 0
```

默认参数保持论文设置：640×640、300 epochs、batch=8、AdamW、lr0=0.001、patience=150、
最后10轮关闭 Mosaic、multi_scale=False、MixUp=0.05、Copy-Paste=0、AMP=False、
seed=42、deterministic=True、workers=2。

说明：论文记录了 `auto_augment=autoaugment`，代码保留该参数以保持参数表一致；但 Ultralytics
8.3.59 的检测训练使用 `v8_transforms`，`auto_augment` 属于分类增强路径，因此不会额外给检测训练
施加 AutoAugment。

## 验证与 APsmall

```bash
python scripts/val.py \
  --weights runs/pdfn/visdrone_pdfn_seed42/weights/best.pt \
  --data configs/datasets/VisDrone.yaml \
  --device 0
```

APsmall：

```bash
python tools/visdrone_to_coco.py \
  --split-dir /data/VisDrone2019-DET-val \
  --output /data/VisDrone2019-DET-val/coco_gt.json

python scripts/eval_coco_apsmall.py \
  --ground-truth /data/VisDrone2019-DET-val/coco_gt.json \
  --predictions runs/detect/val/predictions.json
```

## 表4消融实验

```bash
python scripts/reproduce_all_ablations.py \
  --data configs/datasets/VisDrone.yaml --device 0
```

## 多随机种子稳定性实验

```bash
python scripts/train_multiseed.py \
  --data configs/datasets/VisDrone.yaml \
  --weights yolo11s.pt --device 0
```

实测汇总值：

| Model | mAP@0.5 | mAP@0.5:0.95 | APsmall |
|---|---:|---:|---:|
| YOLO11s | 39.43 ± 0.15 | 23.57 ± 0.12 | 23.77 ± 0.15 |
| PDFN | 42.63 ± 0.15 | 25.80 ± 0.10 | 26.23 ± 0.15 |

复算：

```bash
python scripts/aggregate_multiseed.py \
  --input results_reference/multiseed_measured.csv \
  --output runs/multiseed_summary.csv
```

## Loss 实验

从真实训练目录中的 `results.csv` 复算最后20轮统计值：

```bash
python scripts/analyze_losses.py \
  --input YOLO11s=runs/yolo11s/results.csv \
  --input PDFN=runs/pdfn/results.csv \
  --last 20 --output runs/loss_summary.csv
```

大修实测汇总：YOLO11s 的 val box/cls/dfl loss 为 1.47±0.03、0.89±0.02、0.97±0.02；
PDFN 为 1.40±0.03、0.83±0.02、0.93±0.02。

## 重要说明

`results_reference/` 中只保存论文/大修确认的汇总数值；本项目不会构造不存在的原始日志、
checkpoint 或训练曲线。如果期刊要求原始日志，应另外上传真实保留的 `runs/`、`best.pt/last.pt`
或实验记录，而不能用生成文件替代。
