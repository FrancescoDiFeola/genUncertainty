"""Generic comparison of an original volume with a candidate (analysis A and entry point).

The candidate is a reconstruction in step 1 and a generated volume in step 2: everything
here depends only on the two arrays, the labels and the brain mask, on the same grid and
normalized scale.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DATA_RANGE = 1.0  # Fixed range on the normalized scale, identical for every representation.
RING_MM = 5.0  # Width of the peritumoral ring.


def region_masks(labels: np.ndarray, brain: np.ndarray,
                 ring_mm: float = RING_MM) -> dict[str, np.ndarray]:
    """Boolean masks of the evaluation regions.

    brain: the brain mask; whole_tumor: labels 1-3; enhancing: label 3; ring: the whole
    tumor dilated by `ring_mm` minus the tumor, inside the brain; background: outside the
    brain, evaluated separately because a decoder can leave residue there. Empty regions
    (e.g. no enhancing tumor) are omitted.
    """
    raise NotImplementedError


def global_metrics(original: np.ndarray, candidate: np.ndarray, masks: dict[str, np.ndarray],
                   data_range: float = DATA_RANGE) -> list[dict[str, Any]]:
    """One row per region with keys region, voxels, psnr, ssim, mae, mse.

    SSIM is computed as a full 3D map on the whole volume and then averaged over the
    region, so that windows near a region border still see real neighbors. An exact
    reconstruction gives mse 0 and psnr inf.
    """
    raise NotImplementedError


def compare(original: np.ndarray, candidate: np.ndarray, labels: np.ndarray,
            brain: np.ndarray, *, case_id: str, representation: str,
            analyses: tuple[str, ...] = ("global", "lesions", "frequency"),
            segmentation_original: np.ndarray | None = None,
            segmentation_candidate: np.ndarray | None = None,
            lesion_options: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """All requested tables for one subject and one representation.

    Returns {"global": rows, "lesions": rows, "frequency": rows}; every row carries
    `case_id` and `representation`. Lesion detection needs both segmentations; without
    them the lesion rows still hold sizes and contrast retention. `lesion_options` holds
    min_voxels, shell_mm and detection_fraction for lesions.lesion_table.
    """
    raise NotImplementedError
