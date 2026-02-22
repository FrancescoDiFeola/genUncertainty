import os

def build_paired_list(root_dir):
    """
    Returns a list of dicts:
    [
      {"cbct": ".../cbct_070.npy", "ct": ".../ct_070.npy"},
      ...
    ]
    """
    data = []

    for subject in sorted(os.listdir(root_dir)):
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        cbct_files = sorted(f for f in os.listdir(subject_dir) if f.startswith("cbct_"))

        for cbct_file in cbct_files:
            idx = cbct_file.split("_")[1].split(".")[0]
            ct_file = f"ct_{idx}.npy"

            cbct_path = os.path.join(subject_dir, cbct_file)
            ct_path = os.path.join(subject_dir, ct_file)

            if os.path.exists(ct_path):
                data.append({
                    "cbct": cbct_path,
                    "ct": ct_path,
                })

    return data


def build_single_image_list(root_dir):
    """
    Returns:
    [
      {"img": ".../cbct_070.npy", "modality": "cbct"},
      {"img": ".../ct_070.npy", "modality": "ct"},
      ...
    ]
    """
    data = []

    for subject in sorted(os.listdir(root_dir)):
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        for fname in sorted(os.listdir(subject_dir)):
            if fname.startswith("cbct_") and fname.endswith(".npy"):
                data.append({
                    "img": os.path.join(subject_dir, fname),
                    "modality": "cbct",
                })

            elif fname.startswith("ct_") and fname.endswith(".npy"):
                data.append({
                    "img": os.path.join(subject_dir, fname),
                    "modality": "ct",
                })

    return data

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict


def build_pairs_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    pairs = defaultdict(dict)

    for _, row in df.iterrows():
        img_name = row["img_name"]
        img_path = row["img_path"]
        modality = row["modality"].upper()

        # Example: 1PC010_mr_018
        parts = img_name.split("_")
        subject_id = parts[0]
        slice_id = parts[-1]

        key = (subject_id, slice_id)
        pairs[key][modality] = img_path

    # Keep only complete CBCT–CT pairs
    paired_samples = []
    for (_, _), entry in pairs.items():
        if "CBCT" in entry and "CT" in entry:
            paired_samples.append({
                "cbct_path": entry["CBCT"],
                "ct_path": entry["CT"],
            })

    print(paired_samples)
    return paired_samples

class CBCTCTPaired(Dataset):
    """
    Paired MR–CT dataset reconstructed from a flat CSV.
    Returns:
        A = MR
        B = CT
    """

    def __init__(
        self,
        csv_path: str,
        target_size: int = 512,
        output_size: int = 256,
        ct_min: float = -1000.0,
        ct_max: float = 1000.0,
    ):
        self.samples = build_pairs_from_csv(csv_path)

        self.target_size = target_size
        self.output_size = output_size
        self.ct_min = ct_min
        self.ct_max = ct_max

    def __len__(self):
        return len(self.samples)


    # -------------------------
    # Normalization
    # -------------------------

    def _normalize_ct(self, img):
        img = torch.clamp(img, self.ct_min, self.ct_max)
        img = (img - self.ct_min) / (self.ct_max - self.ct_min)
        return img * 2.0 - 1.0  # [-1, 1]

    # -------------------------
    # Spatial ops (shared)
    # -------------------------
    def _pad_to_target(self, img, pad_value):
        _, h, w = img.shape

        pad_h = max(self.target_size - h, 0)
        pad_w = max(self.target_size - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(
            img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=pad_value,
        )

    def _resize_to_target(self, img, size):
        return F.interpolate(
            img.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _standardize_spatial(self, img, modality):
        _, h, w = img.shape

        # Resize if too large
        if h > self.target_size or w > self.target_size:
            img = self._resize_to_target(img, self.target_size)

        # Pad if too small
        if (h < self.target_size or
                w < self.target_size):
            if modality == "CBCT":
                pad_value = img.min()
            elif modality == "CT":
                pad_value = -1000
            else:
                raise ValueError(f"Unknown modality: {modality}")

            img = self._pad_to_target(img, pad_value)

        return img

    # -------------------------
    # Get item
    # -------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]

        cbct = np.load(sample["cbct_path"]).astype(np.float32)
        ct = np.load(sample["ct_path"]).astype(np.float32)

        if cbct.ndim == 2:
            cbct = cbct[None, ...]
        if ct.ndim == 2:
            ct = ct[None, ...]

        cbct = torch.from_numpy(cbct)
        ct = torch.from_numpy(ct)

        # Shared spatial processing
        cbct = self._standardize_spatial(cbct, modality="CBCT")
        ct = self._standardize_spatial(ct, modality="CT")

        # Normalize
        cbct = self._normalize_ct(cbct)
        ct = self._normalize_ct(ct)

        # Final resize
        cbct = self._resize_to_target(cbct, self.output_size)
        ct = self._resize_to_target(ct, self.output_size)

        return {
            "A": cbct,   # CBCT
            "B": ct,   # CT
        }


