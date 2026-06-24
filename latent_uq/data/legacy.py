from __future__ import annotations

"""Built-in dataset adapters for datasets shipped with the repository.

The public framework API is dataset-agnostic: training and inference code should
receive batches with the standard keys ``condition``, ``target`` and
``case_id``.  This module keeps all historical datasets usable through concise
YAML aliases while avoiding task-specific dataset construction inside the
backend scripts.

Built-in aliases can be used exactly like fully-qualified dataset classes:

    data:
      dataset_class: denoising
      dataset_kwargs:
        annotation_A: /path/to/source.csv
        annotation_B: /path/to/target.csv
      scaling_factor: 7.832608

For new projects, users can instead provide a fully-qualified class path, e.g.
``my_project.datasets.MyDataset``.  No change to this file is required.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from torchvision import transforms

from latent_uq.data.adapters import StandardizedPairedDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRLP_DIR = _REPO_ROOT / "src" / "brlp"


def _load_class(module_filename: str, class_name: str):
    """Load a bundled historical class without importing ``src.brlp`` package.

    Several historical modules rely on top-level imports with side effects.  We
    load the exact file requested by the adapter to keep dataset construction
    isolated and predictable.
    """
    path = _BRLP_DIR / module_filename
    if not path.exists():
        raise FileNotFoundError(f"Cannot find bundled dataset module: {path}")
    module_name = f"latent_uq_builtin_{path.stem}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def _namespace(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


class DenoisingDataset(StandardizedPairedDataset):
    """Low-dose CT / full-dose CT dataset.

    Historical class: ``src/brlp/ldct_hdct_dataset.py::LDCTHDCTDataset``.
    Required kwargs: ``annotation_A``, ``annotation_B``.
    """

    def __init__(self, annotation_A: str, annotation_B: str, **kwargs: Any):
        cls = _load_class("ldct_hdct_dataset.py", "LDCTHDCTDataset")
        super().__init__(cls(annotation_A=annotation_A, annotation_B=annotation_B, **kwargs))


class DenoisingAutoKLDataset(StandardizedPairedDataset):
    """LDCT/HDCT dataset variant historically used for AutoKL/VAE training."""

    def __init__(self, annotation_A: str, annotation_B: str, **kwargs: Any):
        cls = _load_class("ldct_hdct_autoKL_dataset.py", "LDCTHDCTAutoKLDataset")
        super().__init__(cls(annotation_A=annotation_A, annotation_B=annotation_B, **kwargs))


class T1T2Dataset(StandardizedPairedDataset):
    """Paired T1/T2 MRI dataset.

    Historical class: ``src/brlp/T1_T2_dataset.py::T1T2Dataset``.
    Required kwargs: ``annotation_A``, ``annotation_B``.
    """

    def __init__(self, annotation_A: str, annotation_B: str, **kwargs: Any):
        cls = _load_class("T1_T2_dataset.py", "T1T2Dataset")
        super().__init__(cls(annotation_A=annotation_A, annotation_B=annotation_B, **kwargs))


class MotionT1Dataset(StandardizedPairedDataset):
    """Motion-corrupted T1 to clean T1 dataset.

    Historical class: ``src/brlp/motionArtifact_dataset.py::MotionT1Dataset``.
    Required kwargs: ``annotation_A``, ``annotation_B``.
    Useful kwargs: ``mode``, ``motion_range``, ``fixed_motion_level``.
    """

    def __init__(self, annotation_A: str, annotation_B: str, **kwargs: Any):
        cls = _load_class("motionArtifact_dataset.py", "MotionT1Dataset")
        super().__init__(cls(annotation_A=annotation_A, annotation_B=annotation_B, **kwargs))


class CTPETDataset(StandardizedPairedDataset):
    """CT/PET paired dataset from a CSV file.

    Historical class: ``src/brlp/CTPET_dataset.py::CTPETDataset``.
    Required kwargs: ``annotation_A``.
    """

    def __init__(self, annotation_A: str, **kwargs: Any):
        cls = _load_class("CTPET_dataset.py", "CTPETDataset")
        opt = _namespace(annotation_A=annotation_A, **kwargs)
        super().__init__(cls(opt))


class Mri2DSliceDataset(StandardizedPairedDataset):
    """Generic multi-modal MRI 2D slice dataset.

    Historical class: ``src/brlp/Mri2DSlice_dataset.py::Mri2DSlicedataset``.
    Required kwargs are those expected by the historical class, typically
    ``dataroot``, ``mri_modalities``, ``slice_range``, ``phase`` and
    ``under_sample_dataset``.
    """

    def __init__(self, **kwargs: Any):
        cls = _load_class("Mri2DSlice_dataset.py", "Mri2DSlicedataset")
        defaults = dict(
            mri_modalities=["t1n", "t1c", "t2w", "t2f"],
            slice_range=[0, 999],
            phase=None,
            under_sample_dataset=False,
        )
        defaults.update(kwargs)
        super().__init__(cls(_namespace(**defaults)))


class CityscapesDataset(StandardizedPairedDataset):
    """Cityscapes paired RGB/color-label dataset."""

    def __init__(self, root: str, split: str = "train", resize: tuple[int, int] | list[int] | None = (256, 512), **kwargs: Any):
        cls = _load_class("CS_dataset.py", "CityscapesColorDataset")
        transform = None
        if resize is not None:
            transform = transforms.Compose([transforms.Resize(tuple(resize)), transforms.ToTensor()])
        super().__init__(cls(root=root, split=split, transform=transform, target_transform=transform, **kwargs))


class NaturalDenoisingDataset(StandardizedPairedDataset):
    """Generic paired natural-image dataset from the historical ND dataset."""

    def __init__(self, csv_path: str, root_dir: str = "", resize: tuple[int, int] | list[int] | None = (272, 480), **kwargs: Any):
        cls = _load_class("ND_dataset.py", "PairedImageDataset")
        transform = None
        if resize is not None:
            transform = transforms.Compose([transforms.Resize(tuple(resize)), transforms.ToTensor()])
        super().__init__(cls(csv_path=csv_path, root_dir=root_dir, transform_A=transform, transform_B=transform, **kwargs))


class MRCTDataset(StandardizedPairedDataset):
    """MR-to-CT paired dataset."""

    def __init__(self, csv_path: str, **kwargs: Any):
        cls = _load_class("MR_to_CT.py", "MRCTPaired")
        super().__init__(cls(csv_path=csv_path, **kwargs))


class CBCTCTDataset(StandardizedPairedDataset):
    """CBCT-to-CT paired dataset."""

    def __init__(self, csv_path: str, **kwargs: Any):
        cls = _load_class("CBCTtoCT_dataset.py", "CBCTCTPaired")
        super().__init__(cls(csv_path=csv_path, **kwargs))


# Public aliases available through `data.dataset_class`.
# Keep aliases lowercase because `factory.import_object` lowercases lookup keys.
BUILTIN_DATASETS = {
    "denoising": "latent_uq.data.legacy.DenoisingDataset",
    "ldct": "latent_uq.data.legacy.DenoisingDataset",
    "ldct_hdct": "latent_uq.data.legacy.DenoisingDataset",
    "denoising_autokl": "latent_uq.data.legacy.DenoisingAutoKLDataset",
    "ldct_autokl": "latent_uq.data.legacy.DenoisingAutoKLDataset",
    "t1t2": "latent_uq.data.legacy.T1T2Dataset",
    "t1_motion": "latent_uq.data.legacy.MotionT1Dataset",
    "t1motion": "latent_uq.data.legacy.MotionT1Dataset",
    "motion_t1": "latent_uq.data.legacy.MotionT1Dataset",
    "ctpet": "latent_uq.data.legacy.CTPETDataset",
    "mri2d": "latent_uq.data.legacy.Mri2DSliceDataset",
    "mri2d_slice": "latent_uq.data.legacy.Mri2DSliceDataset",
    "t1t2_oasis": "latent_uq.data.legacy.Mri2DSliceDataset",
    "cityscapes": "latent_uq.data.legacy.CityscapesDataset",
    "cs": "latent_uq.data.legacy.CityscapesDataset",
    "nd": "latent_uq.data.legacy.NaturalDenoisingDataset",
    "natural_denoising": "latent_uq.data.legacy.NaturalDenoisingDataset",
    "mrtoct": "latent_uq.data.legacy.MRCTDataset",
    "mr_ct": "latent_uq.data.legacy.MRCTDataset",
    "cbcttoct": "latent_uq.data.legacy.CBCTCTDataset",
    "cbct_ct": "latent_uq.data.legacy.CBCTCTDataset",
}
