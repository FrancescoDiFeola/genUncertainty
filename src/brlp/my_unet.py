from generative.networks.nets import DiffusionModelUNet
import torch.nn as nn
import torch
from monai.networks.blocks.convolutions import Convolution
import torch.nn.functional as F

def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    
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