# Patch-based Training and Inference

This update adds optional patch-based execution without changing the default behavior.

## Training

When enabled, the generic trainer extracts one random paired crop from `condition` and `target` before latent encoding/noising.

```bash
python scripts/train.py \
  --config configs/train/dm_aleatoric.yaml \
  --patch-based \
  --patch-size 128 \
  --patch-pad-value -1
```

YAML equivalent:

```yaml
training:
  patch_based: true
  patch_size: 128
  patch_pad_value: -1.0
```

## Inference

When enabled, the inference entrypoint applies the existing backend to sliding-window patches extracted with MONAI-style tiling logic.

```bash
python scripts/infer.py \
  --config configs/inference/dm_aleatoric.yaml \
  --patch-based \
  --patch-size 128 \
  --patch-overlap 0.25 \
  --patch-pad-value -1
```

YAML equivalent:

```yaml
inference:
  patch_based: true
  patch_size: 128
  patch_overlap: 0.25
  patch_pad_value: -1.0
```

## Notes

- Patching is applied over the last two spatial dimensions.
- For `B,C,H,W`, crops are 2D spatial patches.
- For `B,C,D,H,W`, the full depth is preserved and patches are extracted over `H,W`.
- `patch_pad_value` should match the normalized image background. For images normalized to `[-1, 1]`, use `-1.0`.
- The inference patch mode keeps the current backend API intact. Existing backend functions compute/write metrics for each patch.
