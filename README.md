# Latent-UQ

**A framework for uncertainty-aware diffusion and flow-matching models**

Latent-UQ provides a unified implementation of image-level and latent-space generative models with support for:

- standard generation;
- aleatoric uncertainty estimation;
- uncertainty-guided self-conditioning;
- uncertainty quality evaluation;
- reliability analysis.

The framework is task-agnostic and can be applied to paired image-to-image translation problems such as:

- Low-Dose CT → Full-Dose CT;
- T1 MRI → T2 MRI;
- Motion-Corrupted MRI → Clean MRI;
- CT → PET;
- MR → CT;
- CBCT → CT.

---

# Supported Frameworks

| Framework | Description |
|------------|------------|
| DM | Image-level Diffusion Model |
| FM | Image-level Flow Matching |
| LDM | Latent Diffusion Model |
| LFM | Latent Flow Matching |

# Supported Uncertainty Modes

| Mode | Description |
|--------|------------|
| base | Standard generative model |
| aleatoric | Heteroscedastic uncertainty estimation |
| selfcond | Uncertainty-guided self-conditioning |

---

# Framework Overview

```text
Dataset
   │
   ▼
Condition / Target
   │
   ▼
(Optional) VAE Encoder
   │
   ▼
DM / FM / LDM / LFM
   │
   ├── Prediction
   └── Uncertainty
   │
   ▼
Inference
   │
   ▼
Analysis
```

# Repository Structure

```text
latent_uq/
├── configs/
├── scripts/
│   ├── train.py
│   └── infer.py
├── latent_uq/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── inference/
│   ├── backends/
│   ├── schedulers/
│   └── utils/
├── tests/
└── results/
```

# Quick Start

## Training

```bash
python scripts/train.py --config configs/train/ldm_aleatoric.yaml
```

## Inference

```bash
python scripts/infer.py --config configs/inference/ldm_aleatoric.yaml
```

# Supported Datasets

## LDCT Denoising

```yaml
data:
  dataset_class: denoising
  dataset_kwargs:
    annotation_A: /path/to/lowdose.csv
    annotation_B: /path/to/fulldose.csv
  scaling_factor: 7.832608
```

## T1 → T2 MRI

```yaml
data:
  dataset_class: t1t2
  dataset_kwargs:
    annotation_A: /path/to/t1.csv
    annotation_B: /path/to/t2.csv
  scaling_factor: 9.404202
```

## T1 Motion Correction

```yaml
data:
  dataset_class: t1motion
  dataset_kwargs:
    annotation_A: /path/to/corrupted.csv
    annotation_B: /path/to/clean.csv
  motion_level: 0.15
```

# Adding a New Dataset

```python
class MyDataset(BasePairedDataset):
    def __getitem__(self, idx):
        return {
            "condition": condition_tensor,
            "target": target_tensor,
            "case_id": case_id,
        }
```

```yaml
data:
  dataset_class: my_project.datasets.MyDataset
  dataset_kwargs:
    root: /path/to/data
```

# Changing the Backbone

```yaml
model:
  backbone_class: my_project.models.MyBackbone
  backbone_kwargs:
    in_channels: 6
    out_channels: 3
```

For uncertainty-aware modes:

```python
{
    "prediction": pred,
    "logvar": logvar
}
```

# Changing the VAE

```yaml
model:
  vae_class: my_project.models.VAE3D
  vae_kwargs:
    latent_channels: 4
```

# Inference Analyses

```yaml
analysis:
  metrics: true
  sparsification: true
  spatial_error_correlation: true
  calibration_bins: true
```

## Available Analyses

- metrics → MAE, PSNR, SSIM
- sparsification → AUSE, AURG
- spatial_error_correlation → Pearson, Spearman, AUROC
- calibration_bins → uncertainty-error calibration

# Analysis Compatibility

| Mode | metrics | sparsification | spatial_error_correlation | calibration_bins |
|--------|--------|--------|--------|--------|
| base | ✓ | ✗ | ✗ | ✗ |
| aleatoric | ✓ | ✓ | ✓ | ✓ |
| selfcond | ✓ | ✓ | ✓ | ✓ |

# Output Files

```text
results/metrics.csv
```

Typical columns:

```text
case_id
psnr
ssim
mae
u_mean
u_p95
u_p99
u_top1_mean
u_top5_mean
```

# Common Use Cases

| Task | Recommended Framework | Recommended Mode |
|--------|--------|--------|
| LDCT Denoising | LDM | Aleatoric |
| T1 → T2 | LDM | Aleatoric |
| T1 Motion Correction | LDM | SelfCond |
| Generic Paired Translation | DM/FM/LDM/LFM | Base or Aleatoric |
