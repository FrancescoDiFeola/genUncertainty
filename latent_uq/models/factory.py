from __future__ import annotations

from typing import Any, Tuple

import torch
from monai.networks.nets.autoencoderkl import AutoencoderKL

from latent_uq.frameworks import is_latent_framework, normalize_framework
from latent_uq.utils.imports import import_object
from src.VAE.utils.checkpoints_utils import load_checkpoint
from src.brlp import networks


def _load_state_if_path(module: torch.nn.Module, checkpoint_path: str | None, device: str | torch.device) -> None:
    if not checkpoint_path:
        return
    import os
    if not os.path.exists(str(checkpoint_path)):
        # Historical initializers sometimes accept missing/null checkpoints.
        # For dynamic classes, missing paths should be explicit but not fatal for
        # dry-run or newly initialized inference experiments.
        return
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
        module.load_state_dict(state, strict=False)


def build_autoencoder(device: str, vae_ckpt_dir: str | None, framework: str = "ldm", args: Any | None = None) -> torch.nn.Module | None:
    """Build the VAE used by latent frameworks.

    Extension point:
        model.vae_class: my_package.models.VAE3D
        model.vae_kwargs: {...}

    If no custom VAE is provided, the historical MONAI AutoencoderKL is used.
    Image-level frameworks return None.
    """
    framework = normalize_framework(framework)
    if not is_latent_framework(framework):
        return None

    vae_class = getattr(args, "vae_class", None) if args is not None else None
    vae_kwargs = getattr(args, "vae_kwargs", None) if args is not None else None
    vae_kwargs = vae_kwargs or {}
    if vae_class:
        vae = import_object(vae_class)(**vae_kwargs).to(device)
        _load_state_if_path(vae, vae_ckpt_dir, device)
        vae.eval()
        return vae

    if vae_ckpt_dir is None:
        raise ValueError("vae_ckpt is required for latent frameworks ldm/lfm unless model.vae_class provides an initialized VAE.")

    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(128, 128, 256),
        latent_channels=3,
        num_res_blocks=2,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)
    load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=vae_ckpt_dir, model_name="autoencoder")
    autoencoder.eval()
    return autoencoder


def build_latent_model(args: Any, device: str) -> Tuple[torch.nn.Module, torch.nn.Module | None]:
    """Build the generative model and optional context encoder.

    Extension points:
        model.backbone_class: my_package.models.MyUNet
        model.backbone_kwargs: {...}
        model.context_encoder_class: my_package.models.MyContextEncoder
        model.context_encoder_kwargs: {...}

    Fallback uses the historical src.brlp.networks initializers.
    """
    mode = args.mode

    backbone_class = getattr(args, "backbone_class", None)
    backbone_kwargs = getattr(args, "backbone_kwargs", None) or {}
    if backbone_class:
        model = import_object(backbone_class)(**backbone_kwargs).to(device)
        _load_state_if_path(model, getattr(args, "diff_ckpt", None), device)
    else:
        if mode == "base":
            model = networks.init_ddpm(args.in_ch, args.out_ch, args.diff_ckpt).to(device)
        elif mode == "aleatoric":
            model = networks.init_ddpm_aleatoric(args.in_ch, args.out_ch, args.diff_ckpt).to(device)
        elif mode == "selfcond":
            if not args.context_ckpt:
                raise ValueError("context_ckpt is required for mode='selfcond' unless model.context_encoder_class is configured.")
            model = networks.init_ddpm_aleatoric_two_forward(args.in_ch, args.out_ch, args.diff_ckpt).to(device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    context_encoder = None
    if mode == "selfcond":
        context_class = getattr(args, "context_encoder_class", None)
        context_kwargs = getattr(args, "context_encoder_kwargs", None) or {}
        if context_class:
            context_encoder = import_object(context_class)(**context_kwargs).to(device)
            _load_state_if_path(context_encoder, getattr(args, "context_ckpt", None), device)
        else:
            context_encoder = networks.init_spatial_context_encoder(
                channels=args.spatial_enc_channels,
                cross_attention_dim=args.cross_attention_dim,
                checkpoints_path=args.context_ckpt,
            ).to(device)
    return model, context_encoder
