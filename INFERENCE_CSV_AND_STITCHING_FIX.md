# Inference CSV and stitched-analysis fix

## Modified files

- `latent_uq/backends/adapters/inference_backends.py`
  - creates a dedicated legacy calibration CSV writer for aleatoric metrics;
  - passes separate metrics and calibration writers to image-level legacy backends;
  - closes both primary and companion CSV files safely.
- `scripts/infer.py`
  - forwards the companion calibration writer to full-image legacy inference;
  - reports both metrics and calibration output paths.
- `latent_uq/inference/sliding_window.py`
  - applies all analyses after MONAI stitching;
  - writes the same aleatoric metrics and legacy calibration CSV schemas as full-image inference;
  - keeps TensorBoard visualization at full stitched-image level.

## Resulting aleatoric metrics outputs

Both full-image and `--patch-based` inference now save:

- `metrics_<framework>_<mode>_<task>_ep<epoch>.csv`
- `calibration_bins_<framework>_<mode>_<task>_ep<epoch>.csv`

The calibration file uses the legacy schema:

`Sample,Bin,Unc_mean,Err_mean,Count,Type`
