# Training Generated Preview Update

This update adds an optional TensorBoard preview that runs a short inference pass during training image logging.

Previously, `model_prediction` in TensorBoard was the direct U-Net output used by the loss:

- DM/LDM: predicted noise
- FM/LFM: predicted velocity

This is useful for debugging the training objective, but it is not the generated image. The new option allows TensorBoard to show an actual generated preview for the current mini-batch.

## CLI usage

```bash
python scripts/train.py \
  --config configs/train/dm_aleatoric.yaml \
  --tensorboard \
  --log-generated-preview \
  --preview-steps 25
```

Disable it with:

```bash
--no-log-generated-preview
```

## YAML usage

Add to the `training:` section:

```yaml
training:
  tensorboard: true
  log_generated_preview: true
  preview_steps: 25
```

## Notes

- The preview is only executed when images are logged, currently at the beginning and around the middle of each epoch.
- This can slow down training, especially for latent models or large patch sizes.
- DM/LDM previews use the inference scheduler (`DDIMScheduler`).
- FM/LFM previews use the flow inference scheduler (`RFlowScheduler`).
- The TensorBoard tag changes from `train/model_prediction` to `train/generated_preview/model_prediction` when preview mode is enabled.
