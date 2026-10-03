"""Fixed BraTS segmenter used to measure lesion detection (analysis B).

The same frozen model segments the original volumes and the volumes whose T1ce is replaced
by the candidate, so that detection retention isolates the effect of the representation.

Choice still open:
- a MONAI bundle for BraTS segmentation: runs in the current environment;
- nnU-Net BraTS weights: stronger, but heavy dependencies, better in a separate venv.
If the segmenter saw some test patients during its training, the bias affects original
and candidate alike, but must be declared.
"""
from __future__ import annotations

import numpy as np


class Segmenter:
    """Maps the four BraTS modalities to a BraTS label map on the same grid."""

    def __init__(self, kind: str, weights: str, device: str = "cuda"):
        self.kind, self.weights, self.device = kind, weights, device

    def __call__(self, t1: np.ndarray, t1ce: np.ndarray, t2: np.ndarray,
                 flair: np.ndarray) -> np.ndarray:
        """uint8 X,Y,Z labels (0 background, 1 necrotic core, 2 edema, 3 enhancing).

        Inputs are raw intensities on the original grid; the segmenter applies its own
        preprocessing, which must be the one it was trained with.
        """
        raise NotImplementedError
