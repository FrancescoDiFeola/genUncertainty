"""Analysis B: real enhancing lesions, by size and thickness.

Each connected component of the enhancing tumor (label 3) is a lesion. For each one:
size descriptors, contrast retention against a shell of surrounding tissue, and, when
segmentations are available, detection retention by a fixed segmenter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ENHANCING = 3
MIN_VOXELS = 3  # Smaller components are treated as annotation noise; report how many.
CONNECTIVITY = 26
SHELL_MM = 2.5  # Width of the surrounding-tissue shell for contrast.
DETECTION_FRACTION = 0.10  # Detected if the predicted label covers at least 10% of it.


@dataclass
class Lesion:
    """One connected component and its size descriptors, in mm on a 1 mm grid."""
    index: int
    voxels: int
    volume_mm3: float
    diameter_mm: float  # Equivalent sphere diameter, (6 V / pi)^(1/3).
    thickness_max_mm: float  # 2 x maximum of the inner Euclidean distance transform.
    thickness_median_mm: float  # 2 x median of the inner distance over the component.
    centroid: tuple[float, float, float]


def extract_lesions(labels: np.ndarray, label: int = ENHANCING, min_voxels: int = MIN_VOXELS,
                    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
                    ) -> tuple[list[Lesion], np.ndarray, int]:
    """Connected components of `label` (26-connectivity) with their descriptors.

    Returns the kept lesions, an int32 component map (0 background, k for lesion k), and the
    number of components discarded for having fewer than `min_voxels` voxels.
    """
    raise NotImplementedError


def shell(component: np.ndarray, exclude: np.ndarray, brain: np.ndarray,
          width_mm: float = SHELL_MM) -> np.ndarray:
    """Surrounding tissue: the component dilated by `width_mm`, minus every voxel in
    `exclude` (all tumor labels), restricted to the brain."""
    raise NotImplementedError


def contrast_retention(original: np.ndarray, candidate: np.ndarray, component: np.ndarray,
                       surround: np.ndarray) -> dict[str, float]:
    """Contrast of the lesion against its shell, before and after.

    contrast = mean(lesion) - mean(shell); retention = contrast_candidate / contrast_original
    (1 intact, 0 vanished, negative inverted).

    Keys: contrast_original, contrast_candidate, retention, peak_retention (ratio of the
    95th-percentile intensities inside the lesion), mae (inside the lesion).
    """
    raise NotImplementedError


def detection(component: np.ndarray, segmentation_original: np.ndarray,
              segmentation_candidate: np.ndarray, label: int = ENHANCING,
              fraction: float = DETECTION_FRACTION) -> dict[str, Any]:
    """Whether the fixed segmenter finds the lesion in the original and in the candidate.

    A lesion counts as detected if the predicted `label` covers at least `fraction` of its
    voxels. Detection retention is computed later, over lesions detected in the original.

    Keys: detected_original, detected_candidate (bool), coverage_original,
    coverage_candidate (fractions), dice (between the two predictions of `label` inside
    the lesion dilated by 2 mm).
    """
    raise NotImplementedError


def lesion_table(original: np.ndarray, candidate: np.ndarray, labels: np.ndarray,
                 brain: np.ndarray, segmentation_original: np.ndarray | None = None,
                 segmentation_candidate: np.ndarray | None = None, *,
                 min_voxels: int = MIN_VOXELS, shell_mm: float = SHELL_MM,
                 detection_fraction: float = DETECTION_FRACTION) -> list[dict[str, Any]]:
    """One row per lesion: the Lesion fields, the contrast_retention() keys and, if both
    segmentations are given, the detection() keys."""
    raise NotImplementedError
