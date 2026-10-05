# TRUST


## Configuration



| Field | Values (default) |
|---|---|
| `framework` | `dm`, `fm`, `ldm`, `lfm` |
| `mode` | `selfcond` (TRUST), `aleatoric` (TRUST w/o self-conditioning training), `base` (baseline, ℒ_reg) |
| `data.kwargs.csv_path` | CSV with columns `condition,target,case_id`, listing preprocessed `.npy` arrays (`H,W` or `C,H,W`) |
| `model.autoencoder` | Pretrained VAE (`class_path`, `checkpoint`, `kwargs`); required by `ldm`/`lfm`, with matching `model.latent_channels` and `model.scaling_factor` |
| `model.context_input` | `variance` (default); `prediction_variance` also conditions on the prediction (`ldm` only) |
| `training.epochs` | Total number of epochs (1) |
| `training.lr`, `training.weight_decay` | AdamW (1.5e-5, 0.01) |
| `training.regularization` | λ of the heteroscedastic loss (1e-3) |
| `inference.uncertainty` | `auto` (`propagated` for `aleatoric`/`selfcond`, `none` for `base`), `posthoc`, `none` |
| `inference.samples` | Trajectories for `posthoc` (10 for `metrics`, 4 otherwise) |
| `inference.steps` | Sampling steps (50 DDIM for `dm`/`ldm`, 30 Euler for `fm`/`lfm`) |
| `inference.last_k` | Steps K aggregated by `propagated` (`dm`/`ldm` 10; `fm`/`lfm` 30, `fm` `selfcond` sparsification 10) |
| `inference.decode_samples` | Perturbations decoded by `ldm`/`lfm` (10 `aleatoric`, 20 `selfcond`) |
| `inference.self_conditioning` | `false` runs the ablation w/o test-time conditioning |
| `inference.analyses` | Any of `metrics`, `sparsification`, `calibration`, `uncertainty_summary` |

Unset `inference` fields use the defaults above; each run records the resolved values in
its `run.json`.
