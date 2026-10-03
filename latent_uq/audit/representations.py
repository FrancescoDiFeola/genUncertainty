"""The encode-decode round trips under audit.

Every representation takes a normalized volume in RAS orientation, padded to a multiple of
16 (see io.prepare), and returns a reconstruction of the same shape and scale. The
comparison then happens on the original grid after io.restore.

| Representation | Role |
|---|---|
| Identity       | control: must reproduce the input exactly, validates the pipeline |
| HaarRoundTrip  | exact orthonormal transform: second control, and the wavelet route |
| Downsample     | naive compression with the same factor as MAISI (4 per axis) |
| MaisiVAE       | the pretrained MAISI autoencoder (shared by NV-Generate-MR-Brain) |
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np


class Representation(abc.ABC):
    """A deterministic round trip: normalized X,Y,Z volume in, reconstruction out."""
    name: str = "representation"

    @abc.abstractmethod
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Return the reconstruction of `volume`, same shape, dtype float32."""


class Identity(Representation):
    """Returns the input unchanged; every loss metric must be exactly zero."""
    name = "identity"

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HaarRoundTrip(Representation):
    """Orthonormal 3D Haar transform with `levels` levels, then its inverse.

    Exact up to floating-point error; with two levels the coefficient grid matches MAISI's
    latent grid (4x per axis), with 64 values per position instead of 4.
    """
    name = "haar"

    def __init__(self, levels: int = 2):
        self.levels = levels

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Downsample(Representation):
    """Average-pool by `factor` per axis, then interpolate back with `mode`.

    Same spatial compression as the MAISI latent grid, without learning: the reference
    for hypothesis H3 (does the VAE beat naive compression on small lesions?).
    """
    name = "downsample"

    def __init__(self, factor: int = 4, mode: str = "trilinear"):
        self.factor, self.mode = factor, mode

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MaisiVAE(Representation):
    """MAISI's autoencoder: encode to the 4-channel latent at 1/4 resolution, then decode.

    Built from the `autoencoder_def` section of NV-Generate-CTMR's network config with
    monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi, and loaded
    strictly from `weights`: missing or unexpected keys are an error, because MONAI 1.5.2's
    class may differ from the one the weights were saved with.

    Memory: `num_splits` uses MAISI's exact tensor splitting; a sliding window is not
    offered here because its seams would be attributed to the VAE. `half_precision`
    follows MAISI's default; compare against float32 on a few volumes and declare the
    choice. Encoding uses the posterior mean, not a sample, so the round trip is
    deterministic.
    """
    name = "maisi"

    def __init__(self, network_config: str | Path, weights: str | Path, device: str = "cuda",
                 num_splits: int = 1, half_precision: bool = False):
        self.network_config, self.weights = Path(network_config), Path(weights)
        self.device, self.num_splits, self.half_precision = device, num_splits, half_precision

    def encode(self, volume: np.ndarray) -> np.ndarray:
        """Posterior mean of the latent, 4 x X/4 x Y/4 x Z/4."""
        raise NotImplementedError

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decoded volume, X x Y x Z."""
        raise NotImplementedError

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        raise NotImplementedError


REGISTRY: dict[str, type[Representation]] = {
    "identity": Identity,
    "haar": HaarRoundTrip,
    "downsample": Downsample,
    "maisi": MaisiVAE,
}


def build(spec: dict[str, Any]) -> Representation:
    """Build a representation from a config entry such as {type: downsample, factor: 4}.

    An optional `name` key overrides the default name used in the result tables, so that
    two variants of the same type (e.g. MAISI in float16 and float32) can coexist.
    """
    raise NotImplementedError
