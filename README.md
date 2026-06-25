
# GenUncertainty

**A modular framework for uncertainty-aware diffusion and flow-matching models**

GenUncertainty provides a unified framework for training and evaluating image-level and latent generative models with intrinsic uncertainty estimation and self-conditioning.

## Features

- Image-level Diffusion Models (DM)
- Image-level Flow Matching (FM)
- Latent Diffusion Models (LDM)
- Latent Flow Matching (LFM)
- Base, Aleatoric and Self-Conditioned modes
- Configurable training and inference through YAML
- Dataset-agnostic interface
- Modular inference analyses

---

# Architecture

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
(Optional) VAE
   │
   ▼
Generative Backbone
   │
   ├── Base
   ├── Aleatoric
   └── Self-Conditioned
   │
   ▼
Inference
   │
   ▼
Analysis
   │
   ▼
Results
```

---

# Repository Layout

```text
configs/
    datasets/
    train/
    inference/

scripts/
    train.py
    infer.py

latent_uq/
    data/
    models/
    losses/
    inference/
    backends/
    schedulers/
    utils/

tests/
results/
```

## Folder overview

- **configs/**: experiment configuration.
- **scripts/**: user entry points.
- **latent_uq/data/**: dataset factory and dataset implementations.
- **latent_uq/models/**: backbones, VAE and model builders.
- **latent_uq/losses/**: training losses.
- **latent_uq/inference/**: inference utilities and analyses.
- **latent_uq/backends/**: framework-specific execution logic.
- **tests/**: smoke tests.

---

# Installation

```bash
conda create -n genunc python=3.10
conda activate genunc
pip install -r requirements.txt
pip install -e .
python tests/smoke_imports.py
```

---

# Quick Start

## Train

```bash
python scripts/train.py --config configs/train/ldm_aleatoric.yaml
```

## Infer

```bash
python scripts/infer.py --config configs/inference/ldm_aleatoric.yaml
```

---

# Supported Tasks

## LDCT Denoising

```yaml
data:
  dataset_class: denoising
  dataset_kwargs:
    annotation_A: /path/to/lowdose.csv
    annotation_B: /path/to/fulldose.csv
```

## T1 → T2

```yaml
data:
  dataset_class: t1t2
  dataset_kwargs:
    annotation_A: /path/to/t1.csv
    annotation_B: /path/to/t2.csv
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

---

# Configuration

Three configuration groups are used:

- `configs/datasets/`: reusable dataset templates.
- `configs/train/`: training experiments.
- `configs/inference/`: inference experiments.

---

# Inference Analyses

Enable analyses in the inference YAML:

```yaml
analysis:
  metrics: true
  sparsification: true
  spatial_error_correlation: true
  calibration_bins: true
```

| Analysis | Output |
|----------|--------|
| metrics | MAE, PSNR, SSIM |
| sparsification | AUSE, AURG |
| spatial_error_correlation | Pearson, Spearman, AUROC |
| calibration_bins | uncertainty-error calibration |

---

# Extending the Framework

## Add a Dataset

Create a new dataset in `latent_uq/data/` (or its datasets subfolder if present).

Return:

```python
{
    "condition": condition,
    "target": target,
    "case_id": case_id,
}
```

Then update:

```yaml
data:
  dataset_class: my_project.datasets.MyDataset
```

No changes to train.py or infer.py are required.

## Replace the Backbone

Add the implementation under `latent_uq/models/` and update:

```yaml
model:
  backbone_class: my_project.models.MyBackbone
```

## Replace the VAE

For LDM/LFM:

```yaml
model:
  vae_class: my_project.models.VAE3D
```

## Replace the Loss

```yaml
training:
  loss_class: my_project.losses.MyLoss
```

---

# Outputs

Typical output:

```text
results/
    metrics.csv
    predictions/
```

`metrics.csv` typically contains:

- case_id
- MAE
- PSNR
- SSIM
- u_mean
- u_p95
- u_p99
- u_top1_mean
- u_top5_mean

---

# Design Principles

- Dataset-specific logic belongs only to dataset classes.
- Models are configured through YAML.
- Training and inference pipelines are shared across tasks.
- New datasets, backbones, VAEs and losses should be added without modifying the core scripts.
