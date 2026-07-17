# Rectified-flow loss alignment

The generic trainer now makes the FM/LFM objective explicit and matches the
aleatoric legacy implementation.

For FM/LFM, training uses MONAI `RFlowScheduler` to create the interpolated
state and supervises the velocity field with:

```python
v_target = (target - noise).float()
```

For `mode: aleatoric` and `mode: selfcond`, the loss is:

```python
loss = diff_loss_weight * HeteroscedasticLoss(
    pred_velocity_mean,
    pred_velocity_logvar,
    v_target,
)
```

`HeteroscedasticLoss` retains the legacy clamp, precision-weighted squared
error, log-variance term, and low-variance regularizer.

## Configuration

```yaml
training:
  diff_loss_weight: 1.0
  loss_kwargs:
    min_logvar: -7.0
    reg_weight: 0.001
```

TensorBoard now records both:

- `train/loss_step`: weighted optimization loss;
- `train/loss_unweighted_step`: loss before `diff_loss_weight`.
