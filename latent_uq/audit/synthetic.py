"""Analysis C: synthetic lesions of controlled size and contrast (contrast-size curves).

Lesions are inserted into normalized volumes, in normal-appearing tissue, before encoding,
so contrast is defined on the same scale as every metric. Detectability is model-free:
the Rose criterion on the contrast-to-noise ratio against the local background, because
a BraTS segmenter would not label a sphere inserted only in the T1ce as tumor.
Methodological precedent: task-based evaluation with signal-known-exactly detection
(Li, Zhou, Li, Anastasio, IEEE TMI 2021).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DIAMETERS_MM = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20)
EDGE_SIGMA_VOXELS = 0.5  # Gaussian edge, to mimic partial volume.
MARGIN_MM = 5.0  # Minimum distance from tumor, brain border and other lesions.
ROSE_K = 3.0  # Detectable if the contrast-to-noise ratio is at least this.


@dataclass
class SyntheticLesion:
    """One inserted lesion."""
    center: tuple[int, int, int]
    diameter_mm: float
    contrast: float  # Added intensity at the core, on the normalized scale.
    shape: str = "sphere"  # "sphere" or "ring".
    thickness_mm: float | None = None  # Ring wall thickness; None for spheres.


def calibrate_contrasts(lesion_rows: list[dict[str, Any]], levels: int = 3) -> list[float]:
    """Contrast levels taken from the distribution of real enhancing lesions (analysis B),
    e.g. its quartiles, so that synthetic lesions are as conspicuous as real ones."""
    raise NotImplementedError


def place(brain: np.ndarray, exclude: np.ndarray, lesions: list[SyntheticLesion],
          rng: np.random.Generator, margin_mm: float = MARGIN_MM) -> list[SyntheticLesion]:
    """Random centers inside the brain, at least diameter/2 + margin from `exclude` (the
    dilated tumor), from the brain border and from each other; returns placed copies."""
    raise NotImplementedError


def render(shape: tuple[int, int, int], lesion: SyntheticLesion,
           edge_sigma: float = EDGE_SIGMA_VOXELS) -> np.ndarray:
    """Float32 profile in [0, 1]: a sphere or a ring with a Gaussian-blurred edge."""
    raise NotImplementedError


def insert(volume: np.ndarray, lesions: list[SyntheticLesion]
           ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Add contrast x profile for each lesion; returns the new volume and one boolean core
    mask per lesion (profile above 0.5)."""
    raise NotImplementedError


def measure(with_lesions: np.ndarray, candidate: np.ndarray, masks: list[np.ndarray],
            lesions: list[SyntheticLesion], background_mm: float = MARGIN_MM,
            rose_k: float = ROSE_K) -> list[dict[str, Any]]:
    """One row per lesion, against a local background shell of `background_mm`.

    Keys: the SyntheticLesion fields, contrast_retention, peak_retention, cnr_original,
    cnr_candidate (contrast-to-noise ratios), detected_original, detected_candidate
    (cnr >= rose_k).
    """
    raise NotImplementedError
