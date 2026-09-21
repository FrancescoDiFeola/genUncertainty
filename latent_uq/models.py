"""Built-in U-Net/context encoder and explicit extension contracts."""
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn
from generative.networks.nets import DiffusionModelUNet
from monai.networks.blocks import Convolution

from .utils import finite, import_object


class UNet(DiffusionModelUNet):
    """MONAI U-Net with an optional doubled output split into mean and log variance.

    Self-conditioning uses MONAI cross-attention. Additional MONAI constructor
    options can be set in backbone.kwargs. There is no copied U-Net forward.
    """

    def __init__(self, in_channels=2, out_channels=1, uncertainty=True, context_dim=None, **kwargs):
        options = dict(spatial_dims=2,
                       num_channels=(64, 128, 128, 256),
                       attention_levels=(False, False, True, True),
                       num_res_blocks=2,
                       norm_num_groups=32,
                       num_head_channels=8)
        options.update(kwargs)
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels * (2 if uncertainty else 1),
                         with_conditioning=context_dim is not None,
                         cross_attention_dim=context_dim,
                         **options)
        self.uncertainty = uncertainty
        if uncertainty:
            # Rebuilt (not modified in place), so this consumes the same parameter-init
            # RNG draw that later-built modules, such as the context encoder, expect.
            head = Convolution(spatial_dims=options["spatial_dims"],
                               in_channels=options["num_channels"][0],
                               out_channels=2 * out_channels,
                               strides=1,
                               kernel_size=3,
                               padding=1,
                               conv_only=True)
            for parameter in head.parameters():
                nn.init.zeros_(parameter)
            self.out = nn.Sequential(
                nn.GroupNorm(options["norm_num_groups"], options["num_channels"][0], eps=1e-6),
                nn.SiLU(), head)

    def forward(self, x, timesteps, context=None):
        output = super().forward(x=x, timesteps=timesteps, context=context)
        return output.chunk(2, dim=1) if self.uncertainty else output


class ContextEncoder(nn.Module):
    """Built-in spatial context encoder: normalized variance -> N,1,context_dim tokens."""

    def __init__(self, in_channels, context_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                     nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
                                     nn.Linear(128, context_dim))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, normalized_variance):
        return self.encoder(normalized_variance).unsqueeze(1)


@dataclass
class Models:
    backbone: nn.Module
    context: nn.Module | None = None
    vae: nn.Module | None = None
    uncertainty_decoder: nn.Module | None = None

    def modules(self):
        return self.backbone, self.context, self.vae, self.uncertainty_decoder


def load_weights(module, path):
    """Load an explicit file strictly; support raw state dictionaries and common wrappers."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, Mapping) and key in state:
            state = state[key]
            break
    if not isinstance(state, Mapping):
        raise ValueError(f"Invalid state dictionary: {path}")
    module.load_state_dict({
        key.removeprefix("module."): value
        for key, value in state.items()
    },
                           strict=True)


def build_models(config, *, initialize=True):
    """Build architecture; initialize=False loads all weights from a run checkpoint later."""
    spec = config.model
    channels = spec.latent_channels if config.latent else spec.target_channels
    condition_channels = spec.latent_channels if config.latent else spec.condition_channels

    def build(component, defaults=None):
        kwargs = dict(defaults or {})
        kwargs.update(component.kwargs)
        module = import_object(component.class_path)(**kwargs).to(config.device)
        if initialize and component.checkpoint:
            load_weights(module, component.checkpoint)
        return module

    defaults = dict(in_channels=channels + condition_channels,
                    out_channels=channels,
                    uncertainty=config.mode != "base",
                    context_dim=spec.context_dim if config.mode == "selfcond" else None)
    backbone = build(spec.backbone,
                     defaults if spec.backbone.class_path == "latent_uq.models.UNet" else None)
    context = None
    if config.mode == "selfcond":
        if spec.context_encoder is None:
            if spec.context_tokens != 1:
                raise ValueError(
                    "Built-in ContextEncoder produces one token; provide a custom encoder for other token counts"
                )
            context_channels = channels * (2 if spec.context_input == "prediction_variance" else 1)
            context = ContextEncoder(context_channels, spec.context_dim).to(config.device)
        else:
            context = build(spec.context_encoder)
    vae = None
    if config.latent:
        if initialize and not spec.autoencoder.checkpoint:
            raise ValueError(
                "A pretrained VAE checkpoint is required; this trainer does not optimize the VAE")
        vae = build(spec.autoencoder).eval().requires_grad_(False)
    uncertainty_decoder = build(spec.uncertainty_decoder) if spec.uncertainty_decoder else None
    return Models(backbone, context, vae, uncertainty_decoder)


def predict(models, x, timesteps, config, context=None):
    """Custom forward(x, timesteps, context=None) -> tensor, pair, or named mapping."""
    result = models.backbone(x=x, timesteps=timesteps, context=context)
    if isinstance(result, Mapping):
        prediction, logvar = result.get("prediction"), result.get("logvar")
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        prediction, logvar = result
    else:
        prediction, logvar = result, None
    channels = config.model.latent_channels if config.latent else config.model.target_channels
    expected = (x.shape[0], channels, *x.shape[2:])
    if not torch.is_tensor(prediction) or tuple(prediction.shape) != expected:
        raise ValueError(f"Backbone prediction must have shape {expected}")
    finite(prediction, "prediction")
    if config.mode != "base":
        if not torch.is_tensor(logvar) or logvar.shape != prediction.shape:
            raise ValueError(
                "Uncertainty modes require a per-channel logvar tensor matching prediction")
        finite(logvar, "logvar")
    else:
        logvar = None
    return prediction, logvar


def zero_context(batch_size, config, reference):
    return reference.new_zeros(batch_size, config.model.context_tokens, config.model.context_dim)


def norm_percentile(value, pmin=1, pmax=99):
    """Percentile normalization (clamped to [pmin,pmax]) across channels and pixels, per image."""
    value = value.clone().float()
    normalized = torch.zeros_like(value)
    for index, item in enumerate(value):
        low, high = torch.quantile(item, pmin / 100), torch.quantile(item, pmax / 100)
        normalized[index] = (item.clamp(low, high) - low) / (high - low + 1e-8)
    return normalized


def encode_context(models,
                   logvar,
                   config,
                   prediction=None,
                   *,
                   ablation=False,
                   include_prediction=None):
    value = torch.zeros_like(logvar) if ablation else norm_percentile(logvar.exp())
    concat = config.model.context_input == "prediction_variance" if include_prediction is None else include_prediction
    if concat:
        if prediction is None:
            raise ValueError("prediction_variance context requires the prediction")
        value = torch.cat([norm_percentile(prediction), value], dim=1)
    if ablation:
        # LDM zeros the entire concatenated context, including its prediction channels.
        value = torch.zeros_like(value)
    context = models.context(value)
    expected = (logvar.shape[0], config.model.context_tokens, config.model.context_dim)
    if tuple(context.shape) != expected:
        raise ValueError(f"Context encoder must return {expected}, got {tuple(context.shape)}")
    return finite(context, "context")


@torch.no_grad()
def encode(models, image, config):
    if models.vae is None:
        return image
    if config.model.vae_use_forward:
        result = models.vae(image)
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise ValueError(
                "VAE.forward must return (reconstruction, mean, sigma); set vae_use_forward=False for encode-only VAEs"
            )
        latent = result[1]
    else:
        latent = models.vae.encode(image)
        latent = latent[0] if isinstance(latent, (tuple, list)) else latent
    if not torch.is_tensor(latent) or latent.ndim != 4 or latent.shape[
            1] != config.model.latent_channels or latent.shape[0] != image.shape[0]:
        raise ValueError("VAE.encode must return an N,latent_channels,H,W tensor or (mean, ...)")
    return finite(latent * config.model.scaling_factor, "encoded latent")


def decode(models, latent, config):
    if models.vae is None:
        return latent
    pixels = models.vae.decode(latent / config.model.scaling_factor)
    if not torch.is_tensor(pixels) or pixels.ndim != 4 or pixels.shape[
            1] != config.model.target_channels or pixels.shape[0] != latent.shape[0]:
        raise ValueError("VAE.decode must return an N,target_channels,H,W tensor")
    return finite(pixels, "decoded image")


class LatentUncertaintyDecoder(nn.Module):
    """Optional ldm/aleatoric image-space calibration decoder.

    Decodes a latent log-variance map to pixel space for the auxiliary calibration
    loss in training.py; match upsample_factor to the pretrained VAE's downsampling.
    """

    def __init__(self,
                 latent_channels: int = 4,
                 base_channels: int = 64,
                 out_channels: int = 1,
                 upsample_factor: int = 4):
        super().__init__()
        if upsample_factor < 1 or upsample_factor & (upsample_factor - 1):
            raise ValueError("upsample_factor must be a positive power of two")
        if base_channels < 8 * upsample_factor or base_channels % (8 * upsample_factor):
            raise ValueError("base_channels must be divisible by 8 * upsample_factor for GroupNorm")
        self.encoder = nn.Sequential(nn.Conv2d(latent_channels, base_channels, 3, padding=1),
                                     nn.GroupNorm(8, base_channels), nn.SiLU(),
                                     nn.Conv2d(base_channels, base_channels, 3, padding=1),
                                     nn.GroupNorm(8, base_channels), nn.SiLU())
        self.upsampler = nn.ModuleList()
        ch = base_channels
        scale = upsample_factor
        while scale > 1:
            self.upsampler.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                              nn.Conv2d(ch, ch // 2, 3, padding=1), nn.GroupNorm(8, ch // 2),
                              nn.SiLU()))
            ch = ch // 2
            scale //= 2
        self.final = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(ch, out_channels, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logvar_latent: torch.Tensor) -> torch.Tensor:
        x = self.encoder(logvar_latent)
        for up in self.upsampler:
            x = up(x)
        logvar_img = self.final(x)
        return logvar_img
