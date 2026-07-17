from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F


def normalize_patch_size(patch_size: int | Sequence[int]) -> Tuple[int, int]:
    """Return a 2D patch size tuple ``(height, width)``.

    The framework currently applies patching over the two last spatial axes.
    This supports 2D tensors ``B,C,H,W`` and 3D tensors ``B,C,D,H,W`` by
    cropping only ``H,W`` while preserving the depth dimension.
    """
    if isinstance(patch_size, int):
        return int(patch_size), int(patch_size)
    values = list(patch_size)
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) >= 2:
        return int(values[-2]), int(values[-1])
    raise ValueError(f"Invalid patch_size: {patch_size}")


def _spatial_hw(x: torch.Tensor) -> Tuple[int, int]:
    if x.ndim not in (4, 5):
        raise ValueError(
            f"Patch operations expect B,C,H,W or B,C,D,H,W tensors, got {tuple(x.shape)}"
        )
    return int(x.shape[-2]), int(x.shape[-1])


def pad_to_at_least(
    x: torch.Tensor,
    spatial_size: int | Sequence[int],
    value: float = 0.0,
) -> torch.Tensor:
    """Pad the last two spatial dimensions to at least ``spatial_size``.

    Padding is symmetric whenever possible and uses ``value``. This is useful
    for normalized medical images where the background value is often -1.
    """
    patch_h, patch_w = normalize_patch_size(spatial_size)
    h, w = _spatial_hw(x)
    pad_h = max(patch_h - h, 0)
    pad_w = max(patch_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return x

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # torch.nn.functional.pad pads from the last dimension backwards.
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(x, pad, mode="constant", value=float(value))


def random_crop_pair(
    condition: torch.Tensor,
    target: torch.Tensor,
    patch_size: int | Sequence[int] = 128,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop paired tensors using the same spatial coordinates.

    The crop is applied to the last two spatial dimensions. If either tensor is
    smaller than the requested patch, both tensors are first padded to at least
    ``patch_size`` using ``pad_value``.
    """
    patch_h, patch_w = normalize_patch_size(patch_size)
    condition = pad_to_at_least(condition, (patch_h, patch_w), value=pad_value)
    target = pad_to_at_least(target, (patch_h, patch_w), value=pad_value)

    h = min(condition.shape[-2], target.shape[-2])
    w = min(condition.shape[-1], target.shape[-1])

    # Align shapes defensively if condition/target differ after padding.
    condition = condition[..., :h, :w]
    target = target[..., :h, :w]

    top_max = h - patch_h
    left_max = w - patch_w
    top = int(torch.randint(0, top_max + 1, (1,), device=condition.device).item()) if top_max > 0 else 0
    left = int(torch.randint(0, left_max + 1, (1,), device=condition.device).item()) if left_max > 0 else 0

    condition_patch = condition[..., top : top + patch_h, left : left + patch_w]
    target_patch = target[..., top : top + patch_h, left : left + patch_w]
    return condition_patch, target_patch
