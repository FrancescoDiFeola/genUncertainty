"""Volumes for the audit: raw BraTS intensities, MAISI normalization, orientation, padding.

The cache written by latent_uq.brats.prepare_brats() stores, per subject, the raw int16
modalities cropped to the brain (volumes/<case>.npy, channels in index.json order), the
segmentation (volumes/<case>_seg.npy) and the original shape, affine and crop bounds
(volumes/<case>.json). BraTSVolumeDataset normalizes to [-1, 1] for training; the audit
needs the raw values instead, because MAISI expects its own normalization.

Conventions shared by the whole package:
- arrays are float32 X,Y,Z on the original BraTS grid (1 mm isotropic) unless stated;
- "normalized" means MAISI's MRI scaling, ScaleIntensityRangePercentilesd(lower=0.0,
  upper=99.5, b_min=0.0, b_max=1, clip=False): the 0th and 99.5th percentiles of the
  whole volume map to 0 and 1, without clipping, so bright enhancement may exceed 1;
- representations receive and return normalized volumes in RAS orientation, padded to a
  multiple of 16 voxels; prepare() and restore() convert to and from that layout exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPACING_MM = (1.0, 1.0, 1.0)
LABELS = {"necrotic_core": 1, "edema": 2, "enhancing": 3}  # BraTS 2023 GLI.
MAISI_PERCENTILES = (0.0, 99.5)
MAISI_MULTIPLE = 16  # Dimension constraint documented by NV-Generate-CTMR.


@dataclass
class Volume:
    """One subject on its original grid, with raw intensities."""
    case_id: str
    modality: str
    image: np.ndarray  # float32 X,Y,Z, raw intensities of `modality`.
    labels: np.ndarray  # uint8 X,Y,Z, BraTS labels.
    brain: np.ndarray  # bool X,Y,Z, voxels nonzero in any modality.
    affine: np.ndarray  # float64 4x4, voxel to world.


@dataclass
class Layout:
    """How prepare() reoriented and padded a volume, so that restore() undoes it exactly."""
    orientation: np.ndarray  # nibabel orientation array from the original axes to RAS.
    crop: tuple[slice, ...]  # Slices that remove the padding, in RAS space.
    shape: tuple[int, int, int]  # Original X,Y,Z shape.


def split_subjects(split_file: str | Path, split: str = "test",
                   max_subjects: int | None = None) -> list[str]:
    """Case ids of one split of a fold file, in file order (first `max_subjects` if set).

    Use the patient-level fold files written by scripts/make_patient_splits.py, so that no
    test patient has another timepoint in train.
    """
    raise NotImplementedError


def load_volume(cache_dir: str | Path, case_id: str, modality: str = "t1ce") -> Volume:
    """Raw volume of one modality on the original grid, with labels and brain mask.

    Rebuilds the full X,Y,Z array from the brain crop and its bounds, as
    BraTSVolumeDataset.read() does, but without any intensity normalization. The brain mask
    is the union of nonzero voxels across all modalities, as in the cache.
    """
    raise NotImplementedError


def load_modalities(cache_dir: str | Path, case_id: str) -> dict[str, np.ndarray]:
    """All raw modalities on the original grid, keyed by name (t1, t1ce, t2, t2f), for the
    segmenter, which applies its own preprocessing."""
    raise NotImplementedError


def maisi_range(image: np.ndarray) -> tuple[float, float]:
    """The (low, high) intensities that MAISI's normalization maps to 0 and 1: the 0th and
    99.5th percentiles of the whole volume, background included, as MONAI's
    ScaleIntensityRangePercentiles computes them by default."""
    raise NotImplementedError


def maisi_normalize(image: np.ndarray) -> np.ndarray:
    """MAISI's MRI normalization: (image - low) / (high - low) with maisi_range(image).

    Linear and without clipping, so enhancement brighter than the 99.5th percentile stays
    above 1.
    """
    raise NotImplementedError


def maisi_denormalize(normalized: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """Inverse of maisi_normalize for a given (low, high), to give the segmenter a candidate
    on the raw intensity scale of the original."""
    raise NotImplementedError


def prepare(image: np.ndarray, affine: np.ndarray,
            multiple: int = MAISI_MULTIPLE, pad_value: float = 0.0) -> tuple[np.ndarray, Layout]:
    """Reorient to RAS with the affine and pad every axis to a multiple of `multiple`.

    Padding is background (`pad_value`, 0 after normalization). Returns the prepared volume
    and the Layout that restore() needs.
    """
    raise NotImplementedError


def restore(image: np.ndarray, layout: Layout) -> np.ndarray:
    """Inverse of prepare(): remove the padding and return to the original orientation."""
    raise NotImplementedError
