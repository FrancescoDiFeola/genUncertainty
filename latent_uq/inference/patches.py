from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import torch

from latent_uq.data.patches import normalize_patch_size, pad_to_at_least


def _dense_patch_slices(image_size: tuple[int, int], roi_size: tuple[int, int], overlap: float):
    """Return MONAI-compatible sliding-window slices over H,W.

    Uses MONAI's dense_patch_slices when available and falls back to a small
    local implementation otherwise. The returned slices are over the last two
    dimensions only.
    """
    try:
        from monai.data.utils import dense_patch_slices

        scan_interval = tuple(
            max(1, int(round(r * (1.0 - float(overlap))))) for r in roi_size
        )
        return dense_patch_slices(image_size, roi_size, scan_interval)
    except Exception:
        h, w = image_size
        ph, pw = roi_size
        sh = max(1, int(round(ph * (1.0 - float(overlap)))))
        sw = max(1, int(round(pw * (1.0 - float(overlap)))))

        starts_h = list(range(0, max(h - ph, 0) + 1, sh))
        starts_w = list(range(0, max(w - pw, 0) + 1, sw))
        if starts_h[-1] != h - ph:
            starts_h.append(h - ph)
        if starts_w[-1] != w - pw:
            starts_w.append(w - pw)
        return [(slice(i, i + ph), slice(j, j + pw)) for i in starts_h for j in starts_w]


def run_patchwise_backend_inference(
    *,
    args: Any,
    condition: torch.Tensor,
    target: torch.Tensor,
    run_patch_fn: Callable[[torch.Tensor, torch.Tensor, int], Any],
    step: int,
    patch_size: int | Sequence[int] = 128,
    overlap: float = 0.25,
    pad_value: float = 0.0,
) -> None:
    """Apply an existing backend inference function over sliding-window patches.

    This helper keeps the current backend API intact. It extracts patches with
    MONAI's sliding-window tiling logic and calls ``run_patch_fn`` for each
    patch. Existing backend functions still compute metrics/write CSV rows for
    the patch they receive.

    Notes
    -----
    - This is intended for patch-wise evaluation/logging with the current legacy
      backend functions, which write results internally and do not return a full
      reconstructed image tensor.
    - Patch indices are encoded in the step as ``step * 100000 + patch_idx`` to
      keep TensorBoard/CSV sample identifiers unique.
    """
    roi_h, roi_w = normalize_patch_size(patch_size)
    condition = pad_to_at_least(condition, (roi_h, roi_w), value=pad_value)
    target = pad_to_at_least(target, (roi_h, roi_w), value=pad_value)

    h = min(int(condition.shape[-2]), int(target.shape[-2]))
    w = min(int(condition.shape[-1]), int(target.shape[-1]))
    condition = condition[..., :h, :w]
    target = target[..., :h, :w]

    slices = _dense_patch_slices((h, w), (roi_h, roi_w), overlap=overlap)
    for patch_idx, spatial_slice in enumerate(slices):
        sh, sw = spatial_slice
        condition_patch = condition[..., sh, sw]
        target_patch = target[..., sh, sw]
        patch_step = int(step) * 100000 + patch_idx
        run_patch_fn(condition_patch, target_patch, patch_step)
