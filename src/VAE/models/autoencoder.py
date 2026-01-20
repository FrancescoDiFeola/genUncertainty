import gc
import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution
from monai.networks.blocks.spatialattention import SpatialAttentionBlock
from monai.networks.nets.autoencoderkl import AEKLResBlock, AutoencoderKL
from monai.utils.type_conversion import convert_to_tensor

# Set up logging configuration
logger = logging.getLogger(__name__)

#svuota la cache di cuda
def _empty_cuda_cache(save_mem: bool) -> None:
    if torch.cuda.is_available() and save_mem:
        torch.cuda.empty_cache()
    return


class GroupNorm3D(nn.GroupNorm):
    """
   Custom 3D Group Normalization with optional print_info output.

   Args:
       num_groups: Number of groups for the group norm.
       num_channels: Number of channels for the group norm.
       eps: Epsilon value for numerical stability.
       affine: Whether to use learnable affine parameters, default to `True`.
       norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
       print_info: Whether to print information, default to `False`.
       save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
   """

    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            norm_float16: bool = False,
            print_info: bool = False,
            save_mem: bool = True,
    ):
        super().__init__(num_groups, num_channels, eps, affine)
        self.norm_float16 = norm_float16
        self.print_info = print_info
        self.save_mem = save_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            raise ValueError("GroupNorm3D received None as input!")
        if self.print_info:
            logger.info(f"GroupNorm3D with input shape: {x.shape()}")

        x = super().forward(x)

        if self.norm_float16:
            x = x.to(dtype=torch.float16)

        if self.save_mem:
            _empty_cuda_cache(self.save_mem)
            gc.collect()

        if self.print_info:
            logger.info(f"MaisiGroupNorm3D with output shape: {x.shape()}")

        return x


class Convolution3D(nn.Module):
    """
    A Convolution module that includes Convolution, GroupNorm, ReLU and Dropout.

    Args:
        dimensions: number of spatial dimensions of the input image.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride.
        kernel_size: convolution kernel size.
        dropout_prob: probability of an element to be zeroed.
        norm_groups: number of groups to separate the channels into.
        norm_name: feature normalization type and arguments.
        act_name: activation type and arguments.
        dropout_name: dropout type and arguments.
        bias: whether to have a bias term.
        conv_only: whether to include normalization, activation and dropout.
        print_info: whether to print information.
        save_mem: whether to save GPU memory.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            print_info: bool = False,
            save_mem: bool = True,
            strides: Sequence[int] | int = 1,
            kernel_size: Sequence[int] | int = 3,
            adn_ordering: str = "NDA",
            act: tuple | str | None = "PRELU",
            norm: tuple | str | None = "INSTANCE",
            dropout: tuple | str | float | None = None,
            dropout_dim: int = 1,
            dilation: Sequence[int] | int = 1,
            groups: int = 1,
            bias: bool = True,
            conv_only: bool = False,
            is_transposed: bool = False,
            padding: Sequence[int] | int | None = None,
            output_padding: Sequence[int] | int | None = None,

    ) -> None:

        super().__init__()
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernel_size,
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            dropout=dropout,
            dropout_dim=dropout_dim,
            dilation=dilation,
            groups=groups,
            bias=bias,
            conv_only=conv_only,
            is_transposed=is_transposed,
            padding=padding,
            output_padding=output_padding,
        )
        self.print_info = print_info
        self.save_mem = save_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.print_info:
            logger.info(f"Convolution with input shape: {x.shape}")

        x = self.conv(x)

        if self.save_mem:
            _empty_cuda_cache(self.save_mem)
            gc.collect()

        if self.print_info:
            logger.info(f"Convolution with output shape: {x.shape}")

        return x


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels to the layer.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        print_info: Whether to print information.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            use_convtranspose: bool,
            print_info: bool,
            save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.conv = Convolution3D(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2 if use_convtranspose else 1,
            kernel_size=3,
            padding=1,
            conv_only=True,
            is_transposed=use_convtranspose,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.use_convtranspose = use_convtranspose
        self.save_mem = save_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            x = self.conv(x)
            return x

        x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        _empty_cuda_cache(self.save_mem)
        x = self.conv(x)
        _empty_cuda_cache(self.save_mem)

        return x


class Downsample(nn.Module):
    """Convolution_based downsampling layer.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels to the layer.
        out_channels: Number of output channels to the layer.
        print_info: Whether to print information.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            print_info: bool,
            save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.conv = Convolution3D(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=1,
            conv_only=True,
            print_info=print_info,
            save_mem=save_mem,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    A Residual block consisting of a cascade of 2 convolutions+activation+normalization blockm and a residual connection between input and output.
    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Input channels to the layer.
        norm_num_groups: Number of groups for the group norm layer.
        norm_eps: Epsilon for the normalization.
        out_channels: Number of output channels.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm_num_groups: int,
            norm_eps: float,
            out_channels: int,
            norm_float16: bool,
            print_info: bool,
            save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.save_mem = save_mem

        self.norm1 = GroupNorm3D(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )

        self.conv1 = Convolution3D(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
            print_info=print_info,
            save_mem=save_mem,
        )

        self.norm2 = GroupNorm3D(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.conv2 = Convolution3D(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.nin_shortcut = (
            Convolution3D(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
                print_info=print_info,
                save_mem=save_mem,
            )
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        _empty_cuda_cache(self.save_mem)
        h = F.silu(h)
        _empty_cuda_cache(self.save_mem)
        h = self.conv1(h)
        _empty_cuda_cache(self.save_mem)
        h = self.norm2(h)
        _empty_cuda_cache(self.save_mem)
        h = F.silu(h)
        _empty_cuda_cache(self.save_mem)
        h = self.conv2(h)
        _empty_cuda_cache(self.save_mem)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
            _empty_cuda_cache(self.save_mem)
        out = x + h
        out_tensor: torch.tensor = convert_to_tensor(out)
        return out_tensor


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        num_channels: Sequence of block output channels.
        out_channels: Number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: Number of residual blocks (see AEKLResBlock) per level.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        attention_levels: Indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: If True, use non-local attention block.
        include_fc: whether to include the final linear layer in the attention block. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection in the attention block, default to False.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            num_channels: Sequence[int],
            out_channels: int,
            num_res_blocks: Sequence[int],
            norm_num_groups: int,
            norm_eps: float,
            attention_levels: Sequence[bool],
            norm_float16: bool = False,
            print_info: bool = False,
            save_mem: bool = True,
            with_nonlocal_attn: bool = True,
            include_fc: bool = False,
            use_combined_linear: bool = False,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        # Check if attention_levels and num_channels have the same size
        if len(attention_levels) != len(num_channels):
            raise ValueError("attention_levels and num_channels must have the same size")

        # Check if num_res_blocks and num_channels have the same size
        if len(num_res_blocks) != len(num_channels):
            raise ValueError("num_res_blocks and num_channels must have the same size")
        self.print_info = print_info
        self.save_mem = save_mem

        blocks: list[nn.Module] = []

        blocks.append(
            Convolution3D(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                print_info=print_info,
                save_mem=save_mem,
            )
        )

        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                        norm_float16=norm_float16,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(
                    Downsample(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )

        if with_nonlocal_attn:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                )
            )

            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                )
            )

        blocks.append(
            GroupNorm3D(
                num_groups=norm_num_groups,
                num_channels=num_channels[-1],
                eps=norm_eps,
                affine=True,
                norm_float16=norm_float16,
                print_info=print_info,
                save_mem=save_mem,
            )
        )
        blocks.append(
            Convolution3D(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                print_info=print_info,
                save_mem=save_mem,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("x.shape", x.shape)
        for block in self.blocks:
            x = block(x)
            _empty_cuda_cache(self.save_mem)
        return x


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        num_channels: Sequence of block output channels.
        in_channels: Number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks (see AEKLResBlock) per level.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        attention_levels: Indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: If True, use non-local attention block.
        include_fc: whether to include the final linear layer in the attention block. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection in the attention block, default to False.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
            self,
            spatial_dims: int,
            num_channels: Sequence[int],
            latent_channels: int,
            out_channels: int,
            num_res_blocks: Sequence[int],
            norm_num_groups: int,
            norm_eps: float,
            attention_levels: Sequence[bool],
            norm_float16: bool = False,
            print_info: bool = False,
            save_mem: bool = True,
            with_nonlocal_attn: bool = True,
            include_fc: bool = False,
            use_combined_linear: bool = False,
            use_flash_attention: bool = False,
            use_convtranspose: bool = True,
    ) -> None:
        super().__init__()
        self.save_mem = save_mem

        reversed_block_out_channels = list(reversed(num_channels))
        blocks: list[nn.Module] = []

        blocks.append(
            Convolution3D(
                spatial_dims=spatial_dims,
                in_channels=latent_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                print_info=print_info,
                save_mem=save_mem,
            )
        )
        if with_nonlocal_attn:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]

        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                        norm_float16=norm_float16,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )
            if not is_final_block:
                blocks.append(
                    Upsample(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        use_convtranspose=use_convtranspose,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )
        blocks.append(
            GroupNorm3D(
                num_groups=norm_num_groups,
                num_channels=block_in_ch,
                eps=norm_eps,
                affine=True,
                norm_float16=norm_float16,
                print_info=print_info,
                save_mem=save_mem,
            )
        )
        blocks.append(
            Convolution3D(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                print_info=print_info,
                save_mem=save_mem,
            )
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            # print(f"Passing through block {i}: {block.__class__.__name__}, Input shape: {x.shape}")
            x = block(x)
            # print(f"Output shape after block {i}: {x.shape}")
            _empty_cuda_cache(self.save_mem)
        return x


class Autoencoder(AutoencoderKL):
    """
    AutoencoderKL with custom Encoder and Decoder.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks per level.
        num_channels: Sequence of block output channels.
        attention_levels: Indicate which level from num_channels contain an attention block.
        latent_channels: Number of channels in the latent space.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        with_encoder_nonlocal_attn: If True, use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: If True, use non-local attention block in the decoder.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        use_checkpointing: If True, use activation checkpointing.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_res_blocks: Sequence[int],
            num_channels: Sequence[int],
            attention_levels: Sequence[bool],
            latent_channels: int = 3,
            norm_num_groups: int = 8,
            norm_eps: float = 1e-6,
            with_encoder_nonlocal_attn: bool = False,
            with_decoder_nonlocal_attn: bool = False,
            include_fc: bool = False,
            use_combined_linear: bool = False,
            use_flash_attention: bool = False,
            use_checkpointing: bool = False,
            use_convtranspose: bool = False,
            norm_float16: bool = False,
            print_info: bool = False,
            save_mem: bool = True,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            num_res_blocks,
            num_channels,
            attention_levels,
            latent_channels,
            norm_num_groups,
            norm_eps,
            with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn,
            use_checkpointing,
            use_convtranspose,
            include_fc,
            use_combined_linear,
            use_flash_attention,
        )

        self.encoder: nn.Module = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels * 2,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )

        self.decoder: nn.Module = Decoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            latent_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self, image):

        latent = self.encoder(image)

        mu, log_var = torch.chunk(latent, 2, dim=1)

        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        reconstruction = self.decoder(z)


        return z, mu, log_var, reconstruction


class AutoencoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z, mu, logvar, reconstruction = self.model(x)
        return reconstruction


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        latent = self.model.encoder(x)
        mu, log_var = torch.chunk(latent, 2, dim=1)
        return mu
