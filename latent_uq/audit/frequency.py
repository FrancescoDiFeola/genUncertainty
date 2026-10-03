"""Analysis D: frequency response of a representation.

Fourier shell correlation (FSC), the standard resolution measure of cryo-electron
microscopy (van Heel and Schatz, 2005), between the original and the reconstruction,
shell by shell in spatial frequency; and the power ratio per shell. Read together:
power preserved with low correlation means the decoder adds texture that does not match
the data.
"""
from __future__ import annotations

from typing import Any

import numpy as np

FSC_THRESHOLD = 0.5


def apodize(volume: np.ndarray, mask: np.ndarray, taper_mm: float = 5.0) -> np.ndarray:
    """Multiply by a soft version of `mask` (cosine taper of `taper_mm`) to avoid edge
    artifacts in the Fourier transform; applied identically to both volumes."""
    raise NotImplementedError


def fourier_shell_correlation(a: np.ndarray, b: np.ndarray,
                              spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                              shells: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """FSC between two volumes of equal shape.

    Returns (frequency in cycles/mm at each shell center, correlation per shell). A volume
    with itself gives 1 at every shell.
    """
    raise NotImplementedError


def power_ratio(original: np.ndarray, candidate: np.ndarray,
                spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                shells: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power of `candidate` divided by that of `original`, per shell."""
    raise NotImplementedError


def resolution_mm(frequency: np.ndarray, fsc: np.ndarray,
                  threshold: float = FSC_THRESHOLD) -> float:
    """Spatial scale 1/f (mm) at which the FSC first falls below `threshold`, linearly
    interpolated between shells; inf if it never does."""
    raise NotImplementedError


def frequency_table(original: np.ndarray, candidate: np.ndarray, brain: np.ndarray
                    ) -> list[dict[str, Any]]:
    """One row per shell (frequency, FSC, power ratio) plus the effective resolution."""
    raise NotImplementedError
