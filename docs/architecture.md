# Architecture

Latent-UQ exposes one public interface for four generative frameworks:

- `dm`: image-level diffusion
- `fm`: image-level flow matching
- `ldm`: latent diffusion
- `lfm`: latent flow matching

The same dataset interface is used for all frameworks. Each batch provides a condition image and a target image. Image-level models operate directly on the condition image. Latent models first encode the condition image through the VAE.

```text
Dataset batch
  ├── condition
  └── target
        │
        ├── dm/fm:  condition is used directly
        │
        └── ldm/lfm: condition is encoded by VAE
                    │
                    ▼
           reverse-time model
              ├── base
              ├── aleatoric
              └── selfcond
                    │
                    ▼
           prediction + optional uncertainty
                    │
                    ▼
           metrics / sparsification / calibration
```

## Modes

### `base`

The model predicts only the reverse-time update.

### `aleatoric`

The model predicts the reverse-time update and a log-variance map. The log-variance map is used to estimate intrinsic uncertainty.

### `selfcond`

The model predicts update and uncertainty while conditioning subsequent steps on previous uncertainty estimates through a context encoder.
