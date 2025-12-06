import importlib
from typing import Sequence
from monai.utils import ensure_tuple_rep
from generative.networks.nets import DiffusionModelUNet
import torch.nn as nn
import torch
from monai.networks.blocks.convolutions import Convolution
import torch.nn.functional as F
import math

def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding

class DiffusionUNetWithUncertainty(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        logvar_head_channels: int = 1,
        **unet_kwargs,
    ):
        """
        Wrapper around MONAI's DiffusionModelUNet to add a log-variance head.

        Args:
            in_channels: Input channels (e.g., 2 for [x_t | condition]).
            out_channels: Output channels for noise prediction.
            logvar_head_channels: Channels for log-var prediction (typically 1).
            spatial_dims: 2D or 3D diffusion.
            **unet_kwargs: All other kwargs forwarded to DiffusionModelUNet.
        """
        super().__init__()

        # Create base U-Net
        self.unet = DiffusionModelUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            **unet_kwargs
        )

        # Determine the channel size before final output
        hidden_channels = self.unet.block_out_channels[0]  # Used in final conv

        # Define logvar prediction head (parallel to final output)
        self.logvar_head = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=hidden_channels,
                out_channels=logvar_head_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ):
        """
        Same as DiffusionModelUNet forward, but returns (pred_noise, pred_logvar)

        Returns:
            - pred_noise: output from base U-Net
            - pred_logvar: output from auxiliary head (same size as pred_noise)
        """
        # ==== Copy forward logic ====

        # 1. timestep embedding
        t_emb = get_timestep_embedding(timesteps, self.unet.block_out_channels[0]).to(dtype=x.dtype)
        emb = self.unet.time_embed(t_emb)



        # 2. optional class embedding
        if self.unet.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided for class-conditional model.")
            class_emb = self.unet.class_embedding(class_labels).to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.unet.conv_in(x)

        # 4. down blocks
        down_block_res_samples: list[torch.Tensor] = [h]
        if context is not None and not self.unet.with_conditioning:
            raise ValueError("Model was not initialized with conditioning support.")

        for downsample_block in self.unet.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Optional ControlNet residuals
        if down_block_additional_residuals is not None:
            new_res = []
            for d, r in zip(down_block_res_samples, down_block_additional_residuals):
                new_res.append(d + r)
            down_block_res_samples = tuple(new_res)

        # 5. middle block
        h = self.unet.middle_block(h, temb=emb, context=context)

        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 6. up blocks
        for up_block in self.unet.up_blocks:
            num_res = len(up_block.resnets)
            res = down_block_res_samples[-num_res:]
            down_block_res_samples = down_block_res_samples[:-num_res]
            h = up_block(h, res_hidden_states_list=res, temb=emb, context=context)


        # 7. predict noise
        pred_noise = self.unet.out(h)

        # 8. predict log-variance
        pred_logvar = self.logvar_head(h)

        return (pred_noise, pred_logvar)

class DiffusionModelUNetAleatoricConcat(DiffusionModelUNet):
    def __init__(self, *args, **kwargs):
        # Store the original out_channels
        base_out_channels = kwargs["out_channels"]
        in_channels = kwargs["in_channels"]

        # Set up to double output channels (mean + logvar)
        kwargs["out_channels"] = base_out_channels * 2

        # Call MONAI constructor
        super().__init__(*args, **kwargs)

        # Patch the output layer to match new out_channels
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.block_out_channels[0], eps=1e-6, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=self.conv_in.spatial_dims,
                    in_channels=self.block_out_channels[0],
                    out_channels=kwargs["out_channels"],
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(self, x, timesteps, **kwargs):
        """
        Expects x to be a concatenation of (noisy high-dose | low-dose)
        """
        h = super().forward(x=x, timesteps=timesteps, **kwargs)

        # Split output into predicted noise (mean) and predicted log variance
        pred_mean, pred_logvar = torch.chunk(h, 2, dim=1)

        return (pred_mean, pred_logvar)
        
# Considerations:
# could the chunk order be swapped? Yes, but you must consistently reflect that in training/loss.
# Why predict logvar instead of var? To ensure positivity and numerical stability.
# How do I know itâ€™s logvar? Â Because you apply exp() in the loss, by design.
# In this approach, we let the model learn a data dependent variance.


class MC_Dropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, training=True, inplace=self.inplace)

class FixedMaskDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def reset_mask(self):
        self.mask = None

    def forward(self, x):
        if self.p == 0.0:  #not self.training or
            return x
        if self.mask is None or self.mask.shape != x.shape:
            self.mask = (torch.rand_like(x) > self.p).float() / (1.0 - self.p)
        return x * self.mask
        
class DiffusionModelUNetAleatoricConcat_(DiffusionModelUNet):
    def __init__(self, *args, dropout_rate=0.1, force_dropout=False, **kwargs):
        self.force_dropout = force_dropout
        base_out_channels = kwargs["out_channels"]
        kwargs["out_channels"] = base_out_channels  # We split heads manually

        super().__init__(*args, **kwargs)
        
        self.out = nn.Identity()
        final_feat_dim = self.block_out_channels[0]
        spatial_dims = self.conv_in.spatial_dims

        dropout_cls = nn.Dropout if not self.force_dropout else FixedMaskDropout  # MC_Dropout

        # Dropout (epistemic) only on mean head
        self.mean_head = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=final_feat_dim, eps=1e-6, affine=True),
            nn.SiLU(),
            # dropout_cls(p=dropout_rate),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=final_feat_dim,
                    out_channels=base_out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

        # Aleatoric head (no dropout)
        self.logvar_head = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=final_feat_dim, eps=1e-6, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=final_feat_dim,
                    out_channels=base_out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward_backbone(self, x, timesteps, **kwargs):
        return super().forward(x=x, timesteps=timesteps, **kwargs)

    def forward_mean_head(self, h):
        return self.mean_head(h)

    def forward_logvar_head(self, h):
        return self.logvar_head(h)

    def forward(self, x, timesteps, **kwargs):
        h = self.forward_backbone(x, timesteps, **kwargs)
        pred_mean = self.forward_mean_head(h)
        pred_logvar = self.forward_logvar_head(h)
        return pred_mean, pred_logvar


class DiffusionModelUNetAleatoricEpistemic(DiffusionModelUNet):
    def __init__(self, *args, dropout_rate=0.1, force_dropout=False, **kwargs):
        self.force_dropout = force_dropout
        base_out_channels = kwargs["out_channels"]
        kwargs["out_channels"] = base_out_channels  # We split heads manually

        super().__init__(*args, **kwargs)

        self.out = nn.Identity()
        final_feat_dim = self.block_out_channels[0]
        spatial_dims = self.conv_in.spatial_dims

        dropout_cls = nn.Dropout if not self.force_dropout else FixedMaskDropout

        # Prediction (MAP) head — no dropout
        self.mean_head = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=final_feat_dim, eps=1e-6, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=final_feat_dim,
                    out_channels=base_out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )
        
        # Epistemic uncertainty head — dropout applied
        self.epistemic_head = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=final_feat_dim, eps=1e-6, affine=True),
            nn.SiLU(),
            dropout_cls(p=dropout_rate),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=final_feat_dim,
                    out_channels=base_out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )
        
        # Aleatoric log-variance head — no dropout
        self.logvar_head = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=final_feat_dim, eps=1e-6, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=final_feat_dim,
                    out_channels=base_out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward_backbone(self, x, timesteps, **kwargs):
        return super().forward(x=x, timesteps=timesteps, **kwargs)

    def forward_mean_head(self, h):
        return self.mean_head(h)

    def forward_epistemic_head(self, h):
        return self.epistemic_head(h)

    def forward_logvar_head(self, h):
        return self.logvar_head(h)

    def forward(self, x, timesteps, **kwargs):
        h = self.forward_backbone(x, timesteps, **kwargs)
        pred_mean = self.forward_mean_head(h)
        pred_logvar = self.forward_logvar_head(h)
        return pred_mean, pred_logvar