from __future__ import annotations

from typing import Any

from generative.networks.schedulers import DDIMScheduler

from latent_uq.frameworks import normalize_framework

try:
    from monai.networks.schedulers import RFlowScheduler
except Exception:  # pragma: no cover
    RFlowScheduler = None


def build_scheduler(args: Any, device: str):
    framework = normalize_framework(args.framework)
    if framework in {"ldm", "dm"}:
        return DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            schedule=args.beta_schedule,
            clip_sample=False,
        )
    if framework in {"lfm", "fm"}:
        if RFlowScheduler is None:
            raise RuntimeError("RFlowScheduler is not available in this MONAI installation.")
        scheduler = RFlowScheduler(
            num_train_timesteps=args.num_train_timesteps,
            use_discrete_timesteps=False,
            sample_method="uniform",
            use_timestep_transform=True,
            base_img_size_numel=args.base_img_size_numel,
            spatial_dim=2,
        )
        scheduler.set_timesteps(
            num_inference_steps=args.num_inference_steps,
            device=device,
            input_img_size_numel=args.input_img_size_numel,
        )
        return scheduler
    raise ValueError(f"Unsupported framework: {args.framework}")
