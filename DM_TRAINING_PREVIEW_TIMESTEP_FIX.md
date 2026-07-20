# DM/LDM TensorBoard preview timestep fix

The generated training preview now keeps the two timestep representations required by diffusion inference separate:

- a batch-shaped timestep tensor is passed to the U-Net;
- a scalar timestep is passed to the DDIM scheduler.

This prevents `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` when the preview batch contains more than one sample.

`_scheduler_step(...)` also normalizes repeated timestep tensors defensively before calling scheduler implementations.
