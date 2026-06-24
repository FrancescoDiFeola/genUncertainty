# Configuration Reference

This file documents the YAML fields supported by the public entrypoints:

```bash
python scripts/train.py --config <config.yaml>
python scripts/infer.py --config <config.yaml>
```

---

## `run`

```yaml
run:
  framework: dm        # dm | fm | ldm | lfm
  mode: aleatoric      # base | aleatoric | selfcond
  task: custom         # used for output/checkpoint naming only
  output_dir: outputs
  experiment_name: experiment_name
```

Framework aliases accepted by the CLI:

| Alias | Normalized framework |
|---|---|
| `ddpm`, `diffusion` | `dm` |
| `rf`, `flow`, `flow_matching` | `fm` |

---

## `model`

```yaml
model:
  in_ch: 2
  out_ch: 1
  diff_ckpt: /path/to/model.pth
  vae_ckpt: /path/to/vae_or_dir
  context_ckpt: /path/to/context_encoder.pth

  backbone_class: my_project.models.MyBackbone
  backbone_kwargs: {}

  vae_class: my_project.models.MyVAE
  vae_kwargs: {}

  context_encoder_class: my_project.models.MyContextEncoder
  context_encoder_kwargs: {}
```

Notes:

- `backbone_class` is optional. If omitted, bundled historical network initializers are used.
- `vae_class` is optional. For latent inference without a custom VAE, `vae_ckpt` must point to the historical VAE checkpoint directory.
- `context_encoder_class` is optional. For `selfcond` mode without a custom context encoder, `context_ckpt` is required.

---

## `data`

```yaml
data:
  dataset_class: denoising
  dataset_kwargs:
    annotation_A: /path/to/source.csv
    annotation_B: /path/to/target.csv
  condition_key: condition
  target_key: target
  scaling_factor: 1.0
```

`dataset_class` may be either:

1. a built-in alias, e.g. `denoising`, `t1t2`, `mrtoct`; or
2. a fully-qualified Python class path, e.g. `my_project.datasets.MyDataset`.

The dataset must return either:

```python
{"condition": ..., "target": ..., "case_id": ...}
```

or historical:

```python
{"A": ..., "B": ...}
```

If it uses different keys, set:

```yaml
data:
  condition_key: source
  target_key: label
```

---

## Built-in dataset aliases

| Alias | Dataset | Main required arguments |
|---|---|---|
| `denoising`, `ldct`, `ldct_hdct` | LDCT → HDCT | `annotation_A`, `annotation_B` |
| `denoising_autokl`, `ldct_autokl` | LDCT/HDCT AutoKL variant | `annotation_A`, `annotation_B` |
| `t1t2` | T1 → T2 MRI | `annotation_A`, `annotation_B` |
| `t1motion`, `t1_motion`, `motion_t1` | motion-corrupted T1 → clean T1 | `annotation_A`, `annotation_B` |
| `ctpet` | CT → PET CSV dataset | `annotation_A` |
| `mri2d`, `mri2d_slice`, `t1t2_oasis` | generic multi-modal MRI 2D slice dataset | `dataroot`, `mri_modalities`, `slice_range` |
| `cityscapes`, `cs` | Cityscapes paired dataset | `root`, `split` |
| `nd`, `natural_denoising` | generic paired natural-image dataset | `csv_path`, `root_dir` |
| `mrtoct`, `mr_ct` | MR → CT | `csv_path` |
| `cbcttoct`, `cbct_ct` | CBCT → CT | `csv_path` |

Example:

```yaml
data:
  dataset_class: t1motion
  dataset_kwargs:
    annotation_A: /path/to/t1.csv
    annotation_B: /path/to/t1_clean.csv
    mode: test
    fixed_motion_level: 0.15
  scaling_factor: 9.404202
```

---

## `training`

```yaml
training:
  backend: generic       # generic | legacy
  batch_size: 2
  num_workers: 4
  n_epochs: 1
  lr: 1.5e-5

  loss_class: my_project.losses.MyLoss
  loss_kwargs: {}
```

Recommended setting for new work:

```yaml
training:
  backend: generic
```

Use `backend: legacy` only to reproduce historical task-specific training scripts.

---

## `inference`

```yaml
inference:
  batch_size: 1
  num_workers: 4
  num_train_timesteps: 1000
  num_inference_steps: 50
  beta_start: 0.0015
  beta_end: 0.0205
  beta_schedule: scaled_linear_beta
  K: 10
  seed: 0
```

---

## `analysis`

```yaml
analysis:
  metrics: true
  sparsification: false
  spatial_error_correlation: false
  calibration_bins: false
```

Compatibility:

| Mode | `metrics` | `sparsification` | `spatial_error_correlation` | `calibration_bins` |
|---|---:|---:|---:|---:|
| `base` | yes | no | no | no |
| `aleatoric` | yes | yes | no | no |
| `selfcond` | yes | yes | yes | yes |

Invalid combinations are rejected before inference starts.
