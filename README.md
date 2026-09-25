# GenUncertainty

Conditional diffusion and flow-matching image-to-image models with built-in
uncertainty estimation. Train the same task, for example CT
denoising, MRI motion correction, or cross-modality translation, in pixel
space or in a pretrained VAE's latent space, with three uncertainty-aware
training objectives and several inference-time uncertainty estimators,
through one configuration schema and one CLI.

## Supported models

| Framework | State space | Training target | Sampling |
|---|---|---|---|
| `dm`  | Pixel space | Noise | DDIM |
| `fm`  | Pixel space | Velocity (data − noise) | Rectified flow |
| `ldm` | VAE latent space | Noise | DDIM, then VAE decoding |
| `lfm` | VAE latent space | Velocity (data − noise) | Rectified flow, then VAE decoding |

Every framework supports three training modes:

- **`base`** — mean squared error on the noise/velocity prediction; no uncertainty.
- **`aleatoric`** — the backbone also predicts a log-variance, trained with a heteroscedastic loss.
- **`selfcond`** — as `aleatoric`, plus self-conditioning: the backbone is conditioned, through cross-attention, on the uncertainty it predicted at the previous step.

A single training loop and configuration schema cover every framework/mode
combination. Images are 2D `N,C,H,W` tensors; individual dataset items are `C,H,W`.
`dm` and `fm` also accept 3D `N,C,D,H,W` volumes (see [3D volumes](#3d-volumes)).

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

MONAI `1.5.2` and MONAI Generative `0.2.3` are pinned for scheduler numerical stability.

## Quickstart

```bash
python scripts/make_example.py
python scripts/train.py --config configs/example.yaml --output-dir runs/dm_selfcond
python scripts/infer.py --checkpoint runs/dm_selfcond/checkpoint.pt --output-dir runs/dm_selfcond_eval
python -m pytest -q
```

This trains a small U-Net for one epoch on four synthetic image pairs. It
exercises the full pipeline end to end; it does not download any data and
does not produce a useful model. The reduced backbone size, batch size,
epoch budget and seed are demonstration settings for a fast, CPU-only run.

Once installed, the equivalent commands are:

```bash
latent-uq train --config configs/example.yaml --output-dir runs/experiment
latent-uq infer --checkpoint runs/experiment/checkpoint.pt --output-dir runs/evaluation
```

## Usage

Switch framework and mode with typed overrides:

```bash
latent-uq train --config configs/example.yaml --output-dir runs/fm_base      --set framework=fm --set mode=base
latent-uq train --config configs/example.yaml --output-dir runs/fm_aleatoric --set framework=fm --set mode=aleatoric
latent-uq train --config configs/example.yaml --output-dir runs/fm_selfcond  --set framework=fm --set mode=selfcond
```

`ldm` and `lfm` additionally require a pretrained VAE (see
[Extending the library](#extending-the-library)). Training and inference
reject a nonempty output directory — use a new directory or `--resume`.
`--dry-run` executes a single batch without writing any output, which is
useful for validating a configuration quickly. On CPU, setting
`OMP_NUM_THREADS=1` can speed up small runs.

Repeat `--set section.field=value` for as many fields as needed. Values are
parsed as YAML, so quote lists and strings containing special characters in
the shell. Unknown fields, incompatible settings and wrong scalar types are
rejected at load time.

## Configuration

The full schema and defaults live in [config.py](latent_uq/config.py);
[configs/example.yaml](configs/example.yaml) is a minimal example.

| Section | Purpose |
|---|---|
| `framework`, `mode`, `seed`, `device` | Algorithm, training objective, seed and PyTorch device. |
| `data` | Dataset class, constructor arguments, and `DataLoader` settings. |
| `model` | Backbone, context encoder, pretrained VAE, channel counts and latent scaling. |
| `process` | Diffusion schedule and flow reference sizes. |
| `training` | Epoch budget, optimizer, loss weights, crops, AMP and logging. |
| `inference` | Uncertainty estimator, analyses, sampling counts, tiling and export. |

### Checkpoints

A training run saves `checkpoint.pt`, `config.yaml` and `training.jsonl`.
The checkpoint holds the full model state — including a frozen VAE, if any —
plus the optimizer, AMP scaler, configuration and RNG state, so a run
resumes exactly where it left off:

```bash
latent-uq train --config runs/dm_selfcond/config.yaml --resume runs/dm_selfcond/checkpoint.pt --output-dir runs/dm_selfcond --set training.epochs=2
```

`training.epochs` is always the total epoch budget, not an additional
amount. Resuming rejects any override that would change the trained
algorithm (`framework`, `mode`, `model`, `process`, or most of `training`);
inference may freely override `data`, `inference`, `device` and `seed`.

To evaluate externally trained weights without going through this
project's training loop, set `model.backbone.checkpoint` explicitly
(and `model.context_encoder.checkpoint` for `selfcond`,
`model.autoencoder.checkpoint` for latent models), then:

```bash
latent-uq infer --config my_config.yaml --output-dir runs/evaluation
```

Backbone architecture, channel counts and context architecture must match
the supplied weights — loading is strict. Both raw state dicts and
checkpoints wrapped under `model`, `state_dict` or `model_state_dict` are
accepted, with a leading `module.` prefix stripped automatically.

Optional TensorBoard logging requires `pip install -e '.[logging]'` and
`training.tensorboard=true`. `training.preview_steps` logs a sample
generated with the current weights each epoch, without disturbing the
training random state.

## Training

Diffusion trains with 1,000 timesteps, a `scaled_linear_beta` schedule
(`0.0015`–`0.0205`), noise sampled before timesteps, and a noise-prediction
target. Flow matching uses MONAI's `RFlowScheduler` with continuous
training timesteps and a resolution-dependent timestep transform; its
target is `data − noise`, and its reference resolution defaults to `256²`
in pixel space and `64²` in latent space (`process.flow_base_size` and
related fields override it).

The heteroscedastic loss used by `aleatoric` and `selfcond` is

```text
s = max(predicted_logvar, min_logvar)
loss = mean(0.5 * exp(-s) * (target - prediction)^2 + 0.5 * s) + regularization * mean(exp(-s))
```

with `min_logvar` (default −7) and `regularization` (default 0.001)
configurable under `training`; the clamp applies only inside this loss.
`training.loss_weight` scales the result. The optimizer is AdamW (default
learning rate `1.5e-5`, weight decay `0.01`), with CUDA AMP and no gradient
clipping by default. `data.drop_last` (default `true`) applies only during
training — inference always uses every sample. `data.pin_memory`,
`data.persistent_workers` and `data.prefetch_factor` are passed to the
`DataLoader` (the last two need `data.num_workers > 0`). On CUDA,
`training.jsonl` also records the peak GPU memory of the run so far.

Self-conditioning trains with two forward passes per step: a gradient-free
pass with zero context produces an initial variance estimate, which becomes
the cross-attention context for a second, optimized forward pass on the
same noisy state and timestep. The context encoder is trained jointly with
the backbone, identically for all four frameworks.

## Inference and uncertainty estimation

`inference.uncertainty` selects how predictive uncertainty is estimated:

| Setting | Behavior |
|---|---|
| `auto` | No uncertainty for `base` models; propagated uncertainty otherwise. |
| `none` | Generate the image only; only the `metrics` analysis is available. |
| `propagated` | Accumulate the predicted variance along the sampling trajectory. Requires `aleatoric` or `selfcond`. |
| `posthoc` | Sample several independent trajectories and use their empirical variance. Requires a `base` model. |

Not every combination of mode, uncertainty and analysis is meaningful:

| Mode / estimator | Available analyses |
|---|---|
| `base`, single trajectory | `metrics` |
| `base`, `posthoc` | `metrics`, `sparsification`, `calibration`, `uncertainty_summary` |
| `aleatoric`, propagated | `metrics`, `sparsification` |
| `selfcond`, propagated | `metrics`, `sparsification`, `calibration`, `uncertainty_summary` |
| `selfcond`, ablation (`inference.self_conditioning=false`) | `metrics` |

```bash
latent-uq infer --checkpoint runs/fm_base/checkpoint.pt --output-dir runs/fm_posthoc \
  --set inference.uncertainty=posthoc \
  --set 'inference.analyses=[metrics,sparsification,calibration,uncertainty_summary]'

latent-uq infer --checkpoint runs/dm_selfcond/checkpoint.pt --output-dir runs/dm_ablation \
  --set inference.self_conditioning=false
```

The ablation above keeps the architecture unchanged but feeds a zeroed
context at every step, rather than skipping self-conditioning outright.
Requesting several analyses runs each one **separately**, with its own
sampling pass and its own output subdirectory, since different analyses can
use different sampling parameters. With a single analysis, output files are
written directly in the requested directory.

At inference, self-conditioning reuses the previous step's variance as
context. `propagated` uncertainty accumulates variance over the last
`inference.last_k` steps before the final one; in latent
frameworks, that latent variance reaches pixel space by decoding
`inference.decode_samples` Monte Carlo draws around the final latent state
(the returned image is always the decoded, unperturbed state — only the
variance comes from the draws). `posthoc` uncertainty instead samples
`inference.samples` independent trajectories and reports their population
variance; `sparsification` uses the error of the first draw, the other
analyses use the ensemble mean as the prediction. `inference.steps`
defaults to 50 DDIM steps for diffusion and 30 uniform Euler steps for flow
matching; `last_k`, `samples` and `decode_samples` are similarly nullable,
each with its own model-specific default (recorded in the run's `run.json`).
Leave them unset to use those defaults, or set them explicitly to run a
modified experiment.

### Analyses

- **`metrics`** — MSE, PSNR and SSIM against the target (using each
  target's own intensity range, unless overridden with
  `inference.data_range`), plus, when uncertainty is available,
  Pearson/Spearman correlation and error-detection AUROC (top 5/10/15% of
  pixels) between uncertainty and error, both raw and
  percentile-normalized. For `dm`/`selfcond`, this analysis instead reports
  MAE and the uncertainty-summary statistics below. The CSV header always
  includes every column; whichever a given run does not compute is left
  blank.
- **`sparsification`** — removes pixels in decreasing order of predicted
  uncertainty and tracks the remaining error, against a random-order and
  an oracle (error-ordered) removal curve.
  [`sparsification_scores()`](latent_uq/analysis.py) computes AUSE and
  AURG from the saved curve.
- **`calibration`** — bins pixels by uncertainty percentile
  (`0, 50, 75, 90, 95, 99, 100`) and reports mean uncertainty and mean
  error per bin.
- **`uncertainty_summary`** — mean, 95th/99th percentile, and top 1%/5%
  mean of the uncertainty map.

Reconstruction metrics use the first channel; single-trajectory `base`
inference (no uncertainty) additionally masks target pixels equal to zero.

### Output files

A run writes `config.yaml`, `run.json` and the analysis CSV, plus, unless
disabled, one `predictions/000000.npz` per sample containing `condition`,
`prediction`, `case_id` and, when available, `target` and `variance`.
`run.json` records the resolved sampling settings, the prediction
convention, and the library versions used. Running several analyses adds a
manifest at the root, with each analysis in its own complete subdirectory.

CSV columns use consistent lowercase names and globally increasing sample
IDs; undefined statistics (for example SSIM on a degenerate crop) are left
blank, and an exact reconstruction can report an infinite PSNR.

## Extending the library

### Dataset

[PairedDataset](latent_uq/data.py) reads a CSV of paths to prepared NumPy arrays:

```csv
condition,target,case_id
condition_0.npy,target_0.npy,subject_001
condition_1.npy,target_1.npy,subject_002
```

Arrays are floating-point, shape `H,W` or `C,H,W`, already normalized,
resized and modality-selected — this project does not perform
preprocessing. `case_id` is optional but must be unique if present;
`target` may be omitted for unlabeled inference (set
`inference.analyses: []` to skip error analyses). All rows must agree on
whether targets are present.

Any `torch.utils.data.Dataset` works as a drop-in replacement: return a
dict with `condition`, optional `target`, and optional `case_id`, as
floating-point CHW tensors with matching condition/target spatial
dimensions. Point `data.class_path` at it and pass its constructor
arguments through `data.kwargs` — no change to the training or inference
code is needed.

### 3D volumes

`model.spatial_dims: 3` switches `dm` and `fm` to volumes: the built-in U-Net,
context encoder, flow scheduler, crops and sliding windows become 3D, and
`training.patch_size`/`inference.patch_size` denote cubic patches. Flow matching
additionally requires `process.flow_base_size`, typically the voxel count of
one training patch (`96**3` for 96³ patches), since the 2D default has no 3D
counterpart. Analyses accept `C,D,H,W` volumes unchanged.

### BraTS

[latent_uq/brats.py](latent_uq/brats.py) reads BraTS-style NIfTI releases
described by a fold file — a JSON with a `data_dir` and `train`/`val`/`test`
lists of `{modality: relative path}` entries, one folder per subject. A one-off
conversion (requires `pip install -e '.[brats]'`) stores each subject as a
brain-cropped int16 array plus per-volume intensity percentiles; the same cache
serves every fold:

```bash
python scripts/prepare_brats.py --split-file fold0.json --output-dir /path/to/brats_cache --workers 16
```

`latent_uq.brats.BraTSVolumeDataset` then memory-maps that cache, so a training
patch reads only its own voxels. [configs/brats.yaml](configs/brats.yaml) is a
complete 3D setup for one H200 (96³ patches, batch size 8); set its
`cache_dir` and `split_file`, then:

```bash
latent-uq train --config configs/brats.yaml --output-dir runs/brats_dm_selfcond
latent-uq infer --checkpoint runs/brats_dm_selfcond/checkpoint.pt --output-dir runs/brats_dm_selfcond_test \
  --set data.kwargs.split=test
```

The source and target modalities are `data.kwargs.condition` and
`data.kwargs.target`, in channel order; `model.condition_channels` and
`model.target_channels` must match their lengths. `data.kwargs.patch_size`
crops the train split only, drawing `samples_per_volume` patches per subject
and epoch; other splits yield whole volumes. Each modality is clipped to the
`data.kwargs.percentiles` of its brain voxels and scaled to `[-1, 1]`, with a
background of `-1`.

### Backbone and context encoder

A custom backbone implements `forward(x, timesteps, context=None)`, where
`x` is the noisy target concatenated with the condition along the channel
axis. It returns a prediction tensor for `base` mode, or a
`(prediction, logvar)` pair (or a mapping with `prediction`/`logvar` keys)
for the uncertainty modes. Select it with `model.backbone.class_path`/
`kwargs`; the built-in U-Net accepts any MONAI `DiffusionModelUNet`
constructor option through `kwargs` (the example config uses a small one
for a fast demo). Flash attention is off by default for CPU portability —
enable it explicitly on a compatible GPU.

A custom context encoder returns `N,context_tokens,context_dim`, configured
through `model.context_encoder`. Setting
`model.context_input=prediction_variance` concatenates the normalized
prediction with the normalized variance as context input; only `ldm`
currently supports this — `dm`, `fm` and `lfm` condition on variance alone
and reject that setting.

### Pretrained VAE and calibration decoder

Latent frameworks (`ldm`, `lfm`) require a pretrained VAE — this project
trains the conditional generative model, not the autoencoder:

```yaml
model:
  latent_channels: 3
  scaling_factor: 1.0  # Match the value used to train/normalize this VAE.
  autoencoder:
    class_path: monai.networks.nets.AutoencoderKL
    checkpoint: /path/to/autoencoder.pt
    kwargs:
      spatial_dims: 2
      in_channels: 1
      out_channels: 1
      channels: [128, 128, 256]
      latent_channels: 3
      num_res_blocks: 2
      attention_levels: [false, false, false]
      with_encoder_nonlocal_attn: false
      with_decoder_nonlocal_attn: false
```

`scaling_factor` multiplies the posterior mean before the generative model
and is divided out before decoding. A custom VAE returns
`(reconstruction, mean, sigma)` from `forward()` and implements
`decode(latent)`; `model.vae_use_forward=false` supports encode-only VAEs
whose `encode()` returns a mean tensor (or `(mean, ...)`). The VAE stays
frozen and in evaluation mode throughout training.

`ldm`/`aleatoric` training can optionally add an auxiliary image-space
calibration loss:

```yaml
model:
  uncertainty_decoder:
    class_path: latent_uq.models.LatentUncertaintyDecoder
    kwargs: {latent_channels: 3, out_channels: 1, base_channels: 64, upsample_factor: 4}
training:
  calibration_weight: 0.01
```

The decoder is jointly optimized and saved in the checkpoint; match
`upsample_factor` to the VAE's downsampling factor. It is only available
for `ldm`/`aleatoric`.

### Custom inference analysis

The four built-in analyses cover the common cases; a project-specific one
is added through [CustomAnalysis](latent_uq/analysis.py):

```python
from latent_uq.analysis import CustomAnalysis

class ThresholdedUncertainty(CustomAnalysis):
    columns = ["fraction_above_threshold"]
    requires_uncertainty = True

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def __call__(self, target, prediction, uncertainty):
        return [dict(fraction_above_threshold=float((uncertainty > self.threshold).mean()))]
```

```yaml
inference:
  analyses: [threshold_check]
  custom_analyses:
    threshold_check:
      class_path: my_project.analyses.ThresholdedUncertainty
      kwargs: {threshold: 0.1}
      sampling_analysis: sparsification  # optional; defaults to "metrics"
```

A custom analysis does not influence how its `target`/`prediction`/
`uncertainty` inputs are produced — sampling always runs exactly as for the
built-in analysis named in `sampling_analysis` (`metrics` by default, since
it is the only one available for every framework/mode/uncertainty
combination). `__call__` only defines how to reduce those arrays to CSV
rows, whose columns are declared once in `columns`.

### Python API

```python
from latent_uq.config import from_dict
from latent_uq.models import build_models
from latent_uq.training import load_checkpoint, restore_models
from latent_uq.sampling import sample

checkpoint = load_checkpoint("runs/dm_selfcond/checkpoint.pt")
config = from_dict(checkpoint["config"])
models = build_models(config, initialize=False)
restore_models(models, checkpoint)
# condition is an N,C,H,W tensor on config.device.
result = sample(condition, models, config, analysis="metrics")
image, variance = result.image, result.variance
```

`sample()` takes image-space conditions and handles latent encoding
internally; it disables gradients and restores the model's original
training/evaluation mode. Select one analysis per call, or use the CLI to
run several at once.

`training.patch_size` enables paired random crops during training.
`inference.patch_size` enables MONAI sliding-window inference, with
configurable overlap, blending and window batch size — an extension around
the full-image algorithms above.

### Sliding-window tiling

`inference.tiling` selects how the windows combine into one image:

| Tiling | Behavior |
|---|---|
| `per_window` (default) | Each window runs its own trajectory from its own noise; the final images and variances are blended. |
| `per_window_shared_noise` | As `per_window`, but every window starts from its crop of one image-wide noise draw. |
| `per_step` | One image-wide trajectory: at every step the network is evaluated window by window and its predictions and log variances are blended before the scheduler update; each window's self-conditioning context comes from its crop of the previous step's blended maps. |

The three cost the same number of network evaluations. Under `per_window`,
overlapping windows hold different samples, so blending averages independent
samples there and spatially blended variances ignore cross-window covariance;
under `per_step`, every pixel follows a single trajectory. With a single
window, or with a network whose output at each pixel depends on that pixel
alone, the two shared-noise tilings reproduce whole-image sampling exactly.
Post-hoc variance is computed across independently stitched full images for
every tiling. `per_window_shared_noise` and `per_step` apply to `dm` and `fm`.

## Development

```bash
python -m pytest -q
```

The test suite covers configuration validation, end-to-end training/
inference workflows, checkpoint resume, direct external-weight loading,
tiling, and numerical regression tests for the four core algorithms.
