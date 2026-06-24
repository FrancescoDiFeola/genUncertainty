from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_uq.data.factory import build_dataset
from latent_uq.data.batch import get_condition_target_case_id
from latent_uq.frameworks import is_latent_framework, normalize_framework
from latent_uq.losses.heteroscedastic import HeteroscedasticLoss
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


def _make_training_target(framework: str, target: torch.Tensor):
    b = target.shape[0]
    device = target.device
    noise = torch.randn_like(target)
    if framework in {"dm", "ldm"}:
        timesteps = torch.randint(0, 1000, (b,), device=device).long()
        # Lightweight noising rule for the generic trainer. Project-specific
        # schedulers can be introduced by replacing this function or passing a
        # custom training backend.
        alpha = torch.rand((b, 1, 1, 1), device=device)
        noisy = alpha.sqrt() * target + (1.0 - alpha).sqrt() * noise
        objective = noise
        return noisy, timesteps, objective
    if framework in {"fm", "lfm"}:
        t = torch.rand((b, 1, 1, 1), device=device)
        noisy = (1.0 - t) * noise + t * target
        objective = target - noise
        timesteps = (t.flatten() * 1000).long()
        return noisy, timesteps, objective
    raise ValueError(f"Unsupported framework: {framework}")


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

    params = list(backbone.parameters())
    if context_encoder is not None and bool(getattr(args, "train_context_encoder", False)):
        params += list(context_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    output_dir = Path(getattr(args, "output_dir", "outputs/checkpoints")) / str(getattr(args, "experiment_name", "experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone.train()
    for epoch in range(n_epochs):
        running = 0.0
        for batch in loader:
            condition, target, _ = get_condition_target_case_id(batch)
            condition = condition.to(device).float()
            target = target.to(device).float()

            condition_z = _encode_if_needed(vae, condition, scaling_factor)
            target_z = _encode_if_needed(vae, target, scaling_factor)
            noisy, timesteps, objective = _make_training_target(framework, target_z)
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
            running += float(loss.detach().cpu())

        mean_loss = running / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{n_epochs} - loss: {mean_loss:.6f}")
        torch.save({"model": backbone.state_dict(), "epoch": epoch + 1, "loss": mean_loss}, output_dir / f"model_ep_{epoch + 1}.pth")

    metadata = {
        "framework": framework,
        "mode": mode,
        "dataset_class": getattr(args, "dataset_class", None),
        "scaling_factor": scaling_factor,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
