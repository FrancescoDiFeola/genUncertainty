# Full-image MONAI sliding-window inference

When `--patch-based` is enabled, inference now uses `monai.inferers.sliding_window_inference` and reconstructs a complete prediction before any metric, uncertainty analysis, CSV write, or TensorBoard visualization.

Prediction and uncertainty are concatenated in the predictor output so MONAI applies the same overlap weights to both. Gaussian blending is the default.

Example:

```bash
python3 scripts/infer.py \
  --config configs/inference/fm_aleatoric.yaml \
  --patch-based \
  --patch-size 128 \
  --patch-overlap 0.25 \
  --patch-blend-mode gaussian \
  --patch-sigma-scale 0.125 \
  --patch-pad-value -1
```

TensorBoard tags:

- `Test/SlidingWindow_StitchedInference`
- `Test/stitched_prediction`
- `Test/stitched_uncertainty`

All requested analyses are computed on the stitched full image, not on individual patches.
