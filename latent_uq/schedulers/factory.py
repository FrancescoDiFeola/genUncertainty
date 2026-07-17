from __future__ import annotations

from typing import Any

from latent_uq.frameworks import normalize_framework


def _get(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _import_ddpm_scheduler():
    try:
        from generative.networks.schedulers import DDPMScheduler
        return DDPMScheduler
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "DDPMScheduler is required for dm/ldm training. Install the MONAI "
            "Generative package used by the legacy code, or use a legacy backend."
        ) from exc


def _import_ddim_scheduler():
    try:
        from generative.networks.schedulers import DDIMScheduler
        return DDIMScheduler
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "DDIMScheduler is required for dm/ldm inference. Install the MONAI "
            "Generative package used by the legacy code."
        ) from exc


def _import_rflow_scheduler():
    try:
        from monai.networks.schedulers import RFlowScheduler
        return RFlowScheduler
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "RFlowScheduler is required for fm/lfm training and inference. "
            "Install a MONAI version that provides monai.networks.schedulers.RFlowScheduler."
        ) from exc


def build_diffusion_training_scheduler(args: Any):
    """Build the DDPM scheduler used to generate training noising targets.

    This mirrors the legacy training scripts: DDPMScheduler is used for
    diffusion training, while DDIMScheduler remains the inference sampler.
    """
    DDPMScheduler = _import_ddpm_scheduler()
    return DDPMScheduler(
        num_train_timesteps=int(_get(args, "num_train_timesteps", 1000)),
        beta_start=float(_get(args, "beta_start", 0.0015)),
        beta_end=float(_get(args, "beta_end", 0.0205)),
        schedule=str(_get(args, "beta_schedule", "scaled_linear_beta")),
    )


def build_flow_training_scheduler(args: Any):
    """Build the RFlow scheduler used to generate flow-matching targets."""
    RFlowScheduler = _import_rflow_scheduler()
    return RFlowScheduler(
        num_train_timesteps=int(_get(args, "num_train_timesteps", 1000)),
        use_discrete_timesteps=bool(_get(args, "use_discrete_timesteps", False)),
        sample_method=str(_get(args, "sample_method", "uniform")),
        use_timestep_transform=bool(_get(args, "use_timestep_transform", True)),
        base_img_size_numel=int(_get(args, "base_img_size_numel", 64 * 64)),
        spatial_dim=int(_get(args, "spatial_dim", 2)),
    )


def build_training_scheduler(args: Any):
    """Build the scheduler used during training target construction.

    - dm/ldm: DDPMScheduler, matching legacy diffusion training.
    - fm/lfm: RFlowScheduler, matching legacy flow-matching training.
    """
    framework = normalize_framework(args.framework)
    if framework in {"ldm", "dm"}:
        return build_diffusion_training_scheduler(args)
    if framework in {"lfm", "fm"}:
        return build_flow_training_scheduler(args)
    raise ValueError(f"Unsupported framework: {args.framework}")


def build_scheduler(args: Any, device: str):
    """Build the scheduler used during inference.

    - dm/ldm: DDIMScheduler, matching legacy diffusion inference.
    - fm/lfm: RFlowScheduler with inference timesteps configured.
    """
    framework = normalize_framework(args.framework)
    if framework in {"ldm", "dm"}:
        DDIMScheduler = _import_ddim_scheduler()
        return DDIMScheduler(
            num_train_timesteps=int(_get(args, "num_train_timesteps", 1000)),
            beta_start=float(_get(args, "beta_start", 0.0015)),
            beta_end=float(_get(args, "beta_end", 0.0205)),
            schedule=str(_get(args, "beta_schedule", "scaled_linear_beta")),
            clip_sample=False,
        )
    if framework in {"lfm", "fm"}:
        RFlowScheduler = _import_rflow_scheduler()
        scheduler = RFlowScheduler(
            num_train_timesteps=int(_get(args, "num_train_timesteps", 1000)),
            use_discrete_timesteps=bool(_get(args, "use_discrete_timesteps", False)),
            sample_method=str(_get(args, "sample_method", "uniform")),
            use_timestep_transform=bool(_get(args, "use_timestep_transform", True)),
            base_img_size_numel=int(_get(args, "base_img_size_numel", 64 * 64)),
            spatial_dim=int(_get(args, "spatial_dim", 2)),
        )
        scheduler.set_timesteps(
            num_inference_steps=int(_get(args, "num_inference_steps", 30)),
            device=device,
            input_img_size_numel=int(_get(args, "input_img_size_numel", 64 * 64)),
        )
        return scheduler
    raise ValueError(f"Unsupported framework: {args.framework}")
