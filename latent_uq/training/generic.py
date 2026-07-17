from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.cm as cm
import numpy as np
from latent_uq.data.factory import build_dataset
from latent_uq.data.batch import get_condition_target_case_id
from latent_uq.data.patches import random_crop_pair
from latent_uq.frameworks import is_latent_framework, normalize_framework
from latent_uq.losses.heteroscedastic import HeteroscedasticLoss
from latent_uq.schedulers.factory import build_training_scheduler
from latent_uq.utils.imports import import_object


class IdentityAutoencoder(torch.nn.Module):
    """Fallback autoencoder used when no VAE is configured.

    It keeps image-level and smoke-test configurations executable. For real LDM/LFM
    training, provide `model.vae_class` or use the default MONAI AutoencoderKL
    through the existing VAE options.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def forward(self, x: torch.Tensor):
        return x, x, None


def _cfg_get(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default) if getattr(args, name, None) is not None else default


def _load_checkpoint_if_available(module: torch.nn.Module, checkpoint_path: str | None, device: torch.device) -> None:
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v
    module.load_state_dict(cleaned, strict=False)


def build_backbone(args: Any, device: torch.device) -> torch.nn.Module:
    """Build the generative backbone.

    Preferred extension point:
        model.backbone_class: my_package.models.MyBackbone
        model.backbone_kwargs: {...}

    Fallback:
        historical `src.brlp.networks` initializers.
    """
    backbone_class = getattr(args, "backbone_class", None)
    backbone_kwargs = getattr(args, "backbone_kwargs", None) or {}
    if backbone_class:
        cls = import_object(backbone_class)
        model = cls(**backbone_kwargs).to(device)
        _load_checkpoint_if_available(model, getattr(args, "diff_ckpt", None), device)
        return model

    from src.brlp import networks

    mode = getattr(args, "mode", "base")
    in_ch = int(getattr(args, "in_ch", 2))
    out_ch = int(getattr(args, "out_ch", 1))
    ckpt = getattr(args, "diff_ckpt", None)
    if mode == "base":
        return networks.init_ddpm(in_ch, out_ch, ckpt).to(device)
    if mode == "aleatoric":
        return networks.init_ddpm_aleatoric(in_ch, out_ch, ckpt).to(device)
    if mode == "selfcond":
        return networks.init_ddpm_aleatoric_two_forward(in_ch, out_ch, ckpt).to(device)
    raise ValueError(f"Unsupported mode: {mode}")


def build_context_encoder(args: Any, device: torch.device) -> torch.nn.Module | None:
    if getattr(args, "mode", None) != "selfcond":
        return None
    context_class = getattr(args, "context_encoder_class", None)
    context_kwargs = getattr(args, "context_encoder_kwargs", None) or {}
    if context_class:
        cls = import_object(context_class)
        module = cls(**context_kwargs).to(device)
        _load_checkpoint_if_available(module, getattr(args, "context_ckpt", None), device)
        return module

    from src.brlp import networks

    if not getattr(args, "context_ckpt", None):
        raise ValueError("mode='selfcond' requires model.context_ckpt or model.context_encoder_class.")
    return networks.init_spatial_context_encoder(
        channels=int(getattr(args, "spatial_enc_channels", 1)),
        cross_attention_dim=int(getattr(args, "cross_attention_dim", 128)),
        checkpoints_path=getattr(args, "context_ckpt"),
    ).to(device)


def build_vae(args: Any, device: torch.device) -> torch.nn.Module | None:
    framework = normalize_framework(getattr(args, "framework", "dm"))
    if not is_latent_framework(framework):
        return None

    vae_class = getattr(args, "vae_class", None)
    vae_kwargs = getattr(args, "vae_kwargs", None) or {}
    if vae_class:
        cls = import_object(vae_class)
        vae = cls(**vae_kwargs).to(device)
        _load_checkpoint_if_available(vae, getattr(args, "vae_ckpt", None), device)
        vae.eval()
        return vae

    # Safe default for smoke tests and task-agnostic templates. Real latent runs
    # should provide a VAE class/checkpoint through YAML.
    return IdentityAutoencoder().to(device)


def build_loss(args: Any):
    loss_class = getattr(args, "loss_class", None)
    loss_kwargs = getattr(args, "loss_kwargs", None) or {}
    if loss_class:
        return import_object(loss_class)(**loss_kwargs)
    if getattr(args, "mode", "base") in {"aleatoric", "selfcond"}:
        return HeteroscedasticLoss(**loss_kwargs)
    return torch.nn.MSELoss()


def _encode_if_needed(vae: torch.nn.Module | None, x: torch.Tensor, scaling: float) -> torch.Tensor:
    if vae is None:
        return x
    with torch.no_grad():
        out = vae(x)
        if isinstance(out, tuple):
            z = out[1] if len(out) > 1 else out[0]
        else:
            z = vae.encode(x) if hasattr(vae, "encode") else out
    return z * scaling


def _call_model(model: torch.nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor | None = None):
    try:
        out = model(x=model_input, timesteps=timesteps, context=context)
    except TypeError:
        try:
            out = model(model_input, timesteps, context)
        except TypeError:
            out = model(model_input)
    if isinstance(out, dict):
        pred = out.get("prediction", out.get("pred", None))
        logvar = out.get("logvar", out.get("log_variance", None))
        if pred is None:
            raise ValueError("Model dict output must contain 'prediction'.")
        return pred, logvar
    if isinstance(out, (tuple, list)):
        if len(out) == 1:
            return out[0], None
        return out[0], out[1]
    return out, None


def _sample_timesteps(scheduler: Any, target: torch.Tensor) -> torch.Tensor:
    """Sample training timesteps using the scheduler API when available."""
    if hasattr(scheduler, "sample_timesteps"):
        timesteps = scheduler.sample_timesteps(target)
    else:
        num_train_timesteps = int(getattr(scheduler, "num_train_timesteps", 1000))
        timesteps = torch.randint(
            low=0,
            high=num_train_timesteps,
            size=(target.shape[0],),
            device=target.device,
        )
    return timesteps.to(device=target.device).long()


def _make_diffusion_training_target(target: torch.Tensor, scheduler: Any):
    """Create DDPM training targets with the original DDPMScheduler logic."""
    noise = torch.randn_like(target)
    timesteps = _sample_timesteps(scheduler, target)

    if not hasattr(scheduler, "add_noise"):
        raise AttributeError(
            "Diffusion training requires a scheduler with add_noise(...), "
            "for example generative.networks.schedulers.DDPMScheduler."
        )

    noisy = scheduler.add_noise(
        original_samples=target,
        noise=noise,
        timesteps=timesteps,
    )
    objective = noise
    return noisy, timesteps, objective


def _make_flow_matching_training_target(target: torch.Tensor, scheduler: Any):
    """Create flow-matching training targets with the original RFlowScheduler."""
    noise = torch.randn_like(target)
    timesteps = _sample_timesteps(scheduler, target)

    if not hasattr(scheduler, "add_noise"):
        raise AttributeError(
            "Flow-matching training requires a scheduler with add_noise(...), "
            "for example monai.networks.schedulers.RFlowScheduler."
        )

    noisy = scheduler.add_noise(
        original_samples=target,
        noise=noise,
        timesteps=timesteps,
    )
    objective = target - noise
    return noisy, timesteps, objective


def _make_training_target(framework: str, target: torch.Tensor, scheduler: Any):
    """Create noisy inputs and training objective using the framework scheduler."""
    if framework in {"dm", "ldm"}:
        return _make_diffusion_training_target(target, scheduler)
    if framework in {"fm", "lfm"}:
        return _make_flow_matching_training_target(target, scheduler)
    raise ValueError(f"Unsupported framework: {framework}")

def _build_summary_writer(log_dir: Path, enabled: bool):
    """Create a TensorBoard SummaryWriter lazily.

    This keeps the package importable even when tensorboard is not installed.
    If logging is enabled and tensorboard is missing, a clear runtime error is
    raised with installation instructions.
    """
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "TensorBoard logging is enabled but tensorboard is not installed. "
            "Install it with `pip install tensorboard` or disable logging with "
            "`--no-tensorboard` / `training.tensorboard: false`."
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))

def _as_image_tensor(x: torch.Tensor, max_items: int = 4) -> torch.Tensor:
    """Prepare a tensor for TensorBoard image logging.

    The function is intentionally conservative and task-agnostic:
    - keeps only the first `max_items` samples;
    - if the tensor has more than 3 channels, logs the first channel;
    - normalizes each grid to [0, 1] through `make_grid(normalize=True)`.
    """
    x = x.detach().float().cpu()
    if x.ndim == 5:
        # For 3D tensors B,C,D,H,W, log the central slice.
        x = x[:, :, x.shape[2] // 2]
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected image-like tensor with 3, 4 or 5 dims, got shape {tuple(x.shape)}")
    x = x[:max_items]
    if x.shape[1] not in (1, 3):
        x = x[:, :1]
    return make_grid(x, nrow=min(max_items, x.shape[0]), normalize=True, scale_each=True)

def _as_heatmap_grid(
    x: torch.Tensor,
    max_items: int = 4,
    cmap: str = "inferno",
) -> torch.Tensor:

    """
    Convert a scalar map tensor to an RGB heatmap grid for TensorBoard.
    Accepts:
    - B,C,H,W
    - B,C,D,H,W, using central slice
    Returns:
    - 3,H,W grid
    """

    x = x.detach().float().cpu()
    if x.ndim == 5:
        x = x[:, :, x.shape[2] // 2]

    if x.ndim == 3:
        x = x.unsqueeze(1)

    if x.ndim != 4:
        raise ValueError(f"Expected tensor with 3, 4 or 5 dims, got {tuple(x.shape)}")

    x = x[:max_items]

    if x.shape[1] != 1:
        x = x[:, :1]

    heatmaps = []
    colormap = cm.get_cmap(cmap)
    for i in range(x.shape[0]):
        arr = x[i, 0].numpy()
        p1, p99 = np.percentile(arr, [1, 99])
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1 + 1e-8)
        rgb = colormap(arr)[..., :3]          # H,W,3
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        heatmaps.append(rgb)
    heatmaps = torch.stack(heatmaps, dim=0)

    return make_grid(
        heatmaps,
        nrow=min(max_items, heatmaps.shape[0]),
        normalize=False,
    )

def _log_training_images(
    writer: Any | None,
    *,
    global_step: int,
    condition: torch.Tensor,
    target: torch.Tensor,
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    logvar: torch.Tensor | None = None,
    prefix: str = "train",
    max_items: int = 4,
) -> None:

    if writer is None:
        return
    tensors = {
        "condition": condition,
        "target": target,
        "noisy_or_latent_input": noisy,
        "model_prediction": prediction,
    }

    for name, tensor in tensors.items():
        try:
            writer.add_image(
                f"{prefix}/{name}",
                _as_image_tensor(tensor, max_items=max_items),
                global_step,
            )

        except Exception as exc:
            writer.add_text(
                f"{prefix}/{name}_logging_warning",
                str(exc),
                global_step,
            )

    if logvar is not None:
        try:
            variance = torch.exp(logvar.detach().float())
            writer.add_image(
                f"{prefix}/predicted_variance_heatmap",
                _as_heatmap_grid(
                    variance,
                    max_items=max_items,
                    cmap="inferno",
                ),
                global_step,
            )

        except Exception as exc:

            writer.add_text(
                f"{prefix}/predicted_variance_heatmap_logging_warning",
                str(exc),
                global_step,
            )


def _should_log_images(batch_idx: int, num_batches: int) -> bool:
    """Return True at the beginning and around half epoch."""
    if num_batches <= 1:
        return batch_idx == 0
    half_idx = max(0, num_batches // 2)
    return batch_idx in {0, half_idx}

def run_generic_training(args: Any, cfg: dict[str, Any] | None = None) -> None:
    """Task-agnostic training loop.

    This loop is intentionally minimal and extensible. It supports custom datasets,
    custom backbones, custom VAEs and custom losses via YAML without touching the
    core code. Legacy project-specific training scripts remain available through
    `training.backend: legacy`.
    """
    framework = normalize_framework(getattr(args, "framework", "dm"))
    mode = getattr(args, "mode", "base")
    dry_run = bool(getattr(args, "dry_run", False))

    dataset, scaling_factor = build_dataset(args)
    batch_size = int(getattr(args, "batch_size", 1) or 1)
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    n_epochs = int(getattr(args, "n_epochs", 1) or 1)
    lr = float(getattr(args, "lr", 1e-4) or 1e-4)

    if dry_run:
        print("Generic training dry-run OK")
        print(f"  framework: {framework}")
        print(f"  mode: {mode}")
        print(f"  dataset: {dataset.__class__.__name__}")
        print(f"  batch_size: {batch_size}")
        print(f"  epochs: {n_epochs}")
        return

    device = torch.device(getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    vae = build_vae(args, device)
    backbone = build_backbone(args, device)
    context_encoder = build_context_encoder(args, device)
    criterion = build_loss(args)
    training_scheduler = build_training_scheduler(args)

    params = list(backbone.parameters())
    if context_encoder is not None and bool(getattr(args, "train_context_encoder", False)):
        params += list(context_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    output_dir = Path(getattr(args, "output_dir", "outputs/checkpoints")) / str(getattr(args, "experiment_name", "experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_enabled = bool(getattr(args, "tensorboard", True))
    tensorboard_dir = Path(getattr(args, "tensorboard_dir", None) or (output_dir / "tensorboard"))
    image_log_max_items = int(getattr(args, "image_log_max_items", 4) or 4)

    patch_based = bool(getattr(args, "patch_based", False))
    patch_size = getattr(args, "patch_size", 128)
    patch_pad_value = float(getattr(args, "patch_pad_value", -1.0) if getattr(args, "patch_pad_value", None) is not None else -1.0)

    writer = _build_summary_writer(tensorboard_dir, tensorboard_enabled)
    if writer is not None:
        writer.add_text("config/framework", framework, 0)
        writer.add_text("config/mode", mode, 0)
        writer.add_text("config/dataset", dataset.__class__.__name__, 0)
        writer.add_text("config/patch_based", str(patch_based), 0)
        if patch_based:
            writer.add_text("config/patch_size", str(patch_size), 0)

    global_step = 0
    backbone.train()
    try:
        for epoch in range(n_epochs):
            running = 0.0
            num_batches = max(len(loader), 1)
            for batch_idx, batch in enumerate(loader):
                condition, target, _ = get_condition_target_case_id(batch)
                condition = condition.to(device).float()
                target = target.to(device).float()

                if patch_based:
                    condition, target = random_crop_pair(
                        condition,
                        target,
                        patch_size=patch_size,
                        pad_value=patch_pad_value,
                    )

                condition_z = _encode_if_needed(vae, condition, scaling_factor)
                target_z = _encode_if_needed(vae, target, scaling_factor)
                noisy, timesteps, objective = _make_training_target(framework, target_z, training_scheduler)
                model_input = torch.cat([noisy, condition_z], dim=1)

                context = None
                if mode == "selfcond" and context_encoder is not None:
                    # A neutral context keeps the generic loop executable. For exact
                    # project-specific self-conditioning training, use a specialized
                    # backend or override this block.
                    try:
                        dummy_unc = torch.zeros((condition_z.shape[0], 1, condition_z.shape[-2], condition_z.shape[-1]), device=device)
                        context = context_encoder(dummy_unc)
                    except Exception:
                        context = None

                pred, logvar = _call_model(backbone, model_input, timesteps, context=context)
                if mode in {"aleatoric", "selfcond"} and logvar is not None:
                    loss = criterion(pred, logvar, objective)
                else:
                    loss = criterion(pred, objective)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                running += loss_value
                if writer is not None:
                    writer.add_scalar("train/loss_step", loss_value, global_step)
                    writer.add_scalar("train/epoch_fraction", epoch + (batch_idx + 1) / num_batches, global_step)
                    if logvar is not None:
                        writer.add_scalar("train/logvar_mean", float(logvar.detach().mean().cpu()), global_step)
                        writer.add_scalar("train/variance_mean", float(torch.exp(logvar.detach().float()).mean().cpu()), global_step)
                    if _should_log_images(batch_idx, num_batches):
                        _log_training_images(
                            writer,
                            global_step=global_step,
                            condition=condition,
                            target=target,
                            noisy=noisy,
                            prediction=pred,
                            logvar=logvar,
                            prefix="train",
                            max_items=image_log_max_items,
                        )

                global_step += 1

            mean_loss = running / num_batches
            print(f"Epoch {epoch + 1}/{n_epochs} - loss: {mean_loss:.6f}")
            if writer is not None:
                writer.add_scalar("train/loss_epoch", mean_loss, epoch + 1)
            torch.save({"model": backbone.state_dict(), "epoch": epoch + 1, "loss": mean_loss}, output_dir / f"model_ep_{epoch + 1}.pth")
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    metadata = {
        "framework": framework,
        "mode": mode,
        "dataset_class": getattr(args, "dataset_class", None),
        "scaling_factor": scaling_factor,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
