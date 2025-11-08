import os
from typing import Optional
# from diffusers import UNet2DConditionModel
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from generative.networks.nets import (
    AutoencoderKL,
    PatchDiscriminator,
    DiffusionModelUNet,
    ControlNet
)
from .my_unet import DiffusionModelUNetAleatoricConcat,  DiffusionModelUNetAleatoricEpistemic

"""
def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    
    # Load pretrained weights if available.

    # Args:
    #    checkpoints_path (Optional[str]): path of the checkpoints
    #    network (nn.Module): the neural network to initialize 

    # Returns:
    #    nn.Module: the initialized neural network
    
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        print("Loading checkpoint...")
        network.load_state_dict(torch.load(checkpoints_path))
    return network
"""  

def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), "Invalid path"
        print("Loading checkpoint...")

        # Load the checkpoint
        checkpoint = torch.load(checkpoints_path, map_location=torch.device("cpu"))

        # Remove 'module.' prefix if present
        new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        # Load updated state dict
        network.load_state_dict(new_checkpoint)
    
    return network

def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(spatial_dims=2,
                                in_channels=1,
                                out_channels=1,
                                latent_channels=3,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=2,
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), # Attention in last two levels to enhance global context at deeper layers
                                with_decoder_nonlocal_attn=False,  # Encodes better latent representations with non-local attention in the encoder.
                                with_encoder_nonlocal_attn=False,  # Improves reconstruction quality by enabling attention in the decoder.
                                )  

    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=2,
                                             num_layers_d=3,
                                             num_channels=32,
                                             in_channels=1,
                                             out_channels=1)
    return load_if(checkpoints_path, patch_discriminator)


def init_ddpm_aleatoric(checkpoints_path: Optional[str] = None) -> nn.Module:
    ddpm = DiffusionModelUNetAleatoricConcat(
        spatial_dims=2,  # 2D data (CT slices); use 3 for volumetric 3D CT
        in_channels=2,  # Concatenation of [x_t (noisy high-dose), x_ld (low-dose)] → 2 channels
        out_channels=1,  # Predict noise only; model doubles this to also predict log variance (output = 2 channels)
        num_res_blocks=(2, 2, 2, 2),  # Number of residual blocks at each U-Net level (encoder/decoder depth)
        num_channels=(64, 128, 128, 256),  # Number of feature channels at each U-Net level (controls width of the network)
        attention_levels=(False, False, True, True),  # Whether to use self-attention at each level (deeper levels benefit more)
        norm_num_groups=32,  # Number of groups for GroupNorm (default value used in most LDMs)
        norm_eps=1e-6,  # Epsilon for GroupNorm to avoid divide-by-zero issues
        resblock_updown=False,  # If True, uses residual blocks for up/downsampling (set False for standard blocks)
        num_head_channels=8,  # Number of channels per attention head (used only where attention is enabled)
        with_conditioning=False,  # IMPORTANT: Set to False because you are using spatial concatenation (not cross-attention)
        transformer_num_layers=1,  # Number of layers in the transformer blocks (only applies to attention-enabled levels)
        cross_attention_dim=None,  # Not needed since you're not using cross-attention
        upcast_attention=False,  # Set to True if you want full-precision attention (usually only needed for FP16 training stability)
        use_flash_attention=True,  # Set to True only if using xFormers + GPU to enable flash attention (faster, lower memory)
        dropout_cattn=0.0,  # Dropout in the cross/self-attention layers (typically 0.0 unless overfitting)
        # dropout_rate=0.3, # set the drop-out rate on the last layer
        # force_dropout=True, # We ensure that dropout layers stay active in the mean_head regardless of training/eval mode.
    )
    return load_if(checkpoints_path, ddpm)

def init_ddpm_aleatoric_two_forward(checkpoints_path: Optional[str] = None) -> nn.Module:
    ddpm = DiffusionModelUNetAleatoricConcat(
    		spatial_dims=2,                      # 2D input
    		in_channels=2,                       # x_t and x_ld concatenated
  			out_channels=1,                      # e.g., noise or 2 for noise + logvar
    		num_res_blocks=(2, 2, 2, 2),
   			num_channels=(64, 128, 128, 256),
    		attention_levels=(False, False, True, True),  # Enable attention in deeper levels
    		norm_num_groups=32,
    		norm_eps=1e-6,
    		resblock_updown=False,
    		num_head_channels=8,
    		with_conditioning=True,              # 🔺 Enable conditioning via cross-attention
    		cross_attention_dim=128,            # 🔺 Must match output dim of your encoder
    		transformer_num_layers=1,
    		upcast_attention=False,
   			use_flash_attention=True,
			dropout_cattn=0.0
    )
    return load_if(checkpoints_path, ddpm)

def init_ddpm(checkpoints_path: Optional[str] = None) -> nn.Module:
    ddpm = DiffusionModelUNet(
        spatial_dims=2,  # 2D data (CT slices); use 3 for volumetric 3D CT
        in_channels=2,  # Concatenation of [x_t (noisy high-dose), x_ld (low-dose)] → 2 channels
        out_channels=1,  # Predict noise only; model doubles this to also predict log variance (output = 2 channels)
        num_res_blocks=(2, 2, 2, 2),  # Number of residual blocks at each U-Net level (encoder/decoder depth)
        num_channels=(64, 128, 128, 256),  # Number of feature channels at each U-Net level (controls width of the network)
        attention_levels=(False, False, True, True),  # Whether to use self-attention at each level (deeper levels benefit more)
        norm_num_groups=32,  # Number of groups for GroupNorm (default value used in most LDMs)
        norm_eps=1e-6,  # Epsilon for GroupNorm to avoid divide-by-zero issues
        resblock_updown=False,  # If True, uses residual blocks for up/downsampling (set False for standard blocks)
        num_head_channels=8,  # Number of channels per attention head (used only where attention is enabled)
        with_conditioning=False,  # IMPORTANT: Set to False because you are using spatial concatenation (not cross-attention)
        transformer_num_layers=1,  # Number of layers in the transformer blocks (only applies to attention-enabled levels)
        cross_attention_dim=None,  # Not needed since you're not using cross-attention
        upcast_attention=False,  # Set to True if you want full-precision attention (usually only needed for FP16 training stability)
        use_flash_attention=True,  # Set to True only if using xFormers + GPU to enable flash attention (faster, lower memory)
        dropout_cattn=0.0  # Dropout in the cross/self-attention layers (typically 0.0 unless overfitting)
    )
    return load_if(checkpoints_path, ddpm)

def init_latent_diffusion(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """

    latent_diffusion = DiffusionModelUNet(spatial_dims=2,
                                          in_channels=7,
                                          # ho cambiato il numero di canali a 6 perchè faccio spatial conditioning
                                          out_channels=3,
                                          num_res_blocks=2,
                                          num_channels=(256, 512, 768),
                                          attention_levels=(False, True, True),
                                          norm_num_groups=32,
                                          norm_eps=1e-6,
                                          resblock_updown=True,
                                          num_head_channels=(0, 512, 768),
                                          transformer_num_layers=1,
                                          with_conditioning=False,
                                          # cross_attention_dim=8,
                                          num_class_embeds=None,
                                          upcast_attention=True,
                                          use_flash_attention=False)

    return load_if(checkpoints_path, latent_diffusion)



"""
def init_stable_latent_diffusion(checkpoints_path: Optional[str] = None) -> nn.Module:


    latent_diffusion = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="unet",  # Load only the U-Net component
        torch_dtype=torch.float32  # Use float32 for compatibility
    )
    # Modify the U-Net's first convolution layer to handle 6 input channels
    original_conv = latent_diffusion.conv_in

    # Replace the first convolution with one that accepts 6 input channels
    latent_diffusion.conv_in = nn.Conv2d(6, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding)

    # Initialize the new channels with the weights from the original channels
    with torch.no_grad():
        latent_diffusion.conv_in.weight[:, :4, :, :] = original_conv.weight  # Copy weights for the first 4 channels
        latent_diffusion.conv_in.weight[:, 4:, :, :] = torch.mean(original_conv.weight, dim=1, keepdim=True)  # Average for new channels

    return load_if(checkpoints_path, latent_diffusion)
"""

def init_controlnet(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    controlnet = ControlNet(spatial_dims=3,
                            in_channels=3,
                            num_res_blocks=2,
                            num_channels=(256, 512, 768),
                            attention_levels=(False, True, True),
                            norm_num_groups=32,
                            norm_eps=1e-6,
                            resblock_updown=True,
                            num_head_channels=(0, 512, 768),
                            transformer_num_layers=1,
                            with_conditioning=True,
                            cross_attention_dim=8,
                            num_class_embeds=None,
                            upcast_attention=True,
                            use_flash_attention=False,
                            conditioning_embedding_in_channels=4,
                            conditioning_embedding_num_channels=(256,))
    return load_if(checkpoints_path, controlnet)
    


class SpatialContextEncoder(nn.Module):
    """
    Encodes a spatial 2D map (e.g., uncertainty map) into a global context vector for cross-attention.
    Output shape: (N, 1, cross_attention_dim)
    """

    def __init__(self, in_channels: int, cross_attention_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1. Initial convolution to lift input channels to a richer representation
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            # Motivation: learn low-level spatial features from the uncertainty map; 32 channels are lightweight yet expressive

            nn.ReLU(),
            # Motivation: introduce non-linearity to allow the network to learn complex mappings; avoids dead neurons unlike ReLU6 or hardtanh

            nn.AdaptiveAvgPool2d((4, 4)),
            # Motivation: spatial downsampling to a fixed-size grid (4x4) regardless of input resolution — stabilizes downstream fully connected layers

            nn.Flatten(),
            # Motivation: convert the 4x4x32 tensor to a 1D vector per sample (N, 512), ready for fully connected projection

            nn.Linear(32 * 4 * 4, 128),
            # Motivation: reduce and project to an intermediate latent representation; 128 is a common and expressive size

            nn.ReLU(),
            # Motivation: another non-linearity for representation learning; helps with learning high-level features before final projection

            nn.Linear(128, cross_attention_dim) 
            # Motivation: final projection to match the dimension expected by cross-attention layers in the U-Net (e.g., 128) Even when mapping from 128 to 128, this linear layer can help by learning to better align the feature representation with what the attention mechanism needs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) spatial input (e.g., uncertainty map or auxiliary condition)
        Returns:
            context: (N, 1, cross_attention_dim) for use in cross-attention
        """
        out = self.encoder(x)       # (N, cross_attention_dim)
        return out.unsqueeze(1)    # Motivation: convert to sequence format for attention (1 token per sample)


def init_spatial_context_encoder(channels, checkpoints_path: Optional[str] = None) -> nn.Module:
    spatial_encoder = SpatialContextEncoder(in_channels=channels, cross_attention_dim=128)
    return load_if(checkpoints_path, spatial_encoder)


class RefinerWithCrossAttention(nn.Module):
    def __init__(self, in_channels=1, context_channels=1):
        super().__init__()

        self.query_proj = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.context_proj = nn.Conv2d(context_channels, 64, kernel_size=1)
        self.out_proj = nn.Conv2d(64, in_channels, kernel_size=1)

        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # === Initialization ===
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

    def forward(self, pred_map, context_map):
        # pred_map: (B, C, H, W), context_map: (B, C, H, W)
        B, C, H, W = pred_map.shape

        query = self.query_proj(pred_map).flatten(2).transpose(1, 2)     # (B, HW, 64)
        context = self.context_proj(context_map).flatten(2).transpose(1, 2)  # (B, HW, 64)

        attn_out, _ = self.attn(query, context, context)  # Cross-attention

        attn_out = attn_out.transpose(1, 2).view(B, 64, H, W)
        refined = self.out_proj(attn_out) + pred_map  # Residual refinement
        return refined


class FiLMRefiner(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # 🔸 Feature extraction from the predicted image
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 🔸 Generate gamma and beta from uncertainty map (FiLM conditioning)
        self.gamma_beta_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # Global average pooling -> (B, C, 1, 1)
            nn.Flatten(),                      # -> (B, C)
            nn.Linear(in_channels, 64 * 2)     # -> (B, 128) -> gamma (64), beta (64)
        )

        # 🔸 Output projection back to image space
        self.out = nn.Conv2d(64, in_channels, kernel_size=1)

        # 🔸 Initialize all weights
        self._init_weights()

    def _init_weights(self):
        # Initialize all convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, uncertainty_map):
        """
        Args:
            x: Tensor of shape (B, C, H, W) — predicted image (MAP)
            uncertainty_map: Tensor of shape (B, C, H, W) — uncertainty signal (e.g., predicted log-var)

        Returns:
            Refined prediction of same shape as `x`
        """

        # 🔸 Extract features from input prediction
        feat = self.feature_proj(x)  # (B, 64, H, W)

        # 🔸 Generate FiLM parameters from the uncertainty map
        gamma_beta = self.gamma_beta_gen(uncertainty_map)  # (B, 128)
        gamma, beta = gamma_beta.chunk(2, dim=1)           # Each (B, 64)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)          # (B, 64, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)            # (B, 64, 1, 1)

        # 🔸 Apply FiLM modulation: channel-wise affine transform
        modulated = gamma * feat + beta  # (B, 64, H, W)

        # 🔸 Project back to original space + residual refinement
        out = self.out(modulated) + x  # (B, C, H, W)

        return out

class JointFiLMRefiner(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Feature extractors
        self.mean_feat = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.var_feat = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        # FiLM generators
        self.mean_to_var_film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)  # 64 gamma, 64 beta
        )
        self.var_to_mean_film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

        # Projection heads
        self.mean_out = nn.Conv2d(64, in_channels, kernel_size=1)
        self.var_out = nn.Conv2d(64, in_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, pred_mean, log_var):
        """
        pred_mean: (B, C, H, W)
        log_var:   (B, C, H, W)
        """
        # Feature extraction
        mean_feat = self.mean_feat(pred_mean)  # (B, 64, H, W)
        var_feat = self.var_feat(log_var)      # (B, 64, H, W)

        # FiLM from mean → refine var
        gamma_beta_mv = self.mean_to_var_film(mean_feat)
        gamma_mv, beta_mv = gamma_beta_mv.chunk(2, dim=1)
        gamma_mv = gamma_mv.unsqueeze(-1).unsqueeze(-1)
        beta_mv = beta_mv.unsqueeze(-1).unsqueeze(-1)
        mod_var_feat = gamma_mv * var_feat + beta_mv

        # FiLM from var → refine mean
        gamma_beta_vm = self.var_to_mean_film(var_feat)
        gamma_vm, beta_vm = gamma_beta_vm.chunk(2, dim=1)
        gamma_vm = gamma_vm.unsqueeze(-1).unsqueeze(-1)
        beta_vm = beta_vm.unsqueeze(-1).unsqueeze(-1)
        mod_mean_feat = gamma_vm * mean_feat + beta_vm

        # Output projections (residual)
        refined_mean = self.mean_out(mod_mean_feat) + pred_mean
        refined_logvar = self.var_out(mod_var_feat) + log_var

        return refined_mean, refined_logvar

def init_refiner(checkpoints_path: Optional[str] = None) -> nn.Module:
	refiner = JointFiLMRefiner()
	return load_if(checkpoints_path, refiner)