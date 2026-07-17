# Scheduler Update

This update aligns the generic training loop with the original scheduler logic used in the legacy scripts.

## Changed files

- `latent_uq/schedulers/factory.py`
  - Added `build_training_scheduler(args)`.
  - `dm` / `ldm` training now uses `generative.networks.schedulers.DDPMScheduler`.
  - `dm` / `ldm` inference continues to use `generative.networks.schedulers.DDIMScheduler`.
  - `fm` / `lfm` training and inference use `monai.networks.schedulers.RFlowScheduler`.
  - Scheduler imports are lazy, so dry-runs and smoke imports do not fail on machines without the full training dependencies.

- `latent_uq/training/generic.py`
  - Removed the simplified random-alpha diffusion noising rule.
  - Replaced it with scheduler-based target construction:
    - `DDPMScheduler.add_noise(...)` for `dm` / `ldm`;
    - `RFlowScheduler.add_noise(...)` for `fm` / `lfm`.
  - Training objectives remain:
    - diffusion: `noise`;
    - flow matching: `target - noise`, matching the legacy training scripts.

- `configs/train/*.yaml`
  - Added explicit scheduler parameters to the training section.

- `CONFIG_REFERENCE.md`
  - Documented the training/inference scheduler choices and YAML fields.

## Validation performed

The following checks were run in the packaging environment:

```bash
python -m compileall scripts latent_uq tests
python tests/smoke_imports.py
python scripts/train.py --config configs/train/dm_aleatoric.yaml --dry-run
python scripts/train.py --config configs/train/fm_aleatoric.yaml --dry-run
python scripts/infer.py --config configs/inference/dm_aleatoric.yaml --dry-run
```

Full training requires the same scheduler dependencies as the legacy code: MONAI Generative for `DDPMScheduler` / `DDIMScheduler` and MONAI with `RFlowScheduler` for flow matching.
