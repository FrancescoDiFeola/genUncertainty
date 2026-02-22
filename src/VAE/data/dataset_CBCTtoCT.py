import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class CBCTCTSingleImageDataset(Dataset):
    """
    Single-image MR / CT dataset with:
    - modality-aware normalization
    - spatial standardization to 512x512
    """

    def __init__(
        self,
        csv_path: str,
        target_size: int = 512,
        output_size: int = 256,
        ct_norm: str = "minmax",
        ct_min: float = -1000.0,
        ct_max: float = 1000.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.target_size = target_size

        self.ct_norm = ct_norm
        self.ct_min = ct_min
        self.ct_max = ct_max
        self.output_size = output_size

    def __len__(self):
        return len(self.df)

    # -------------------------
    # Normalization
    # -------------------------

    def _normalize_ct(self, img):
        img = np.clip(img, self.ct_min, self.ct_max)
        img = (img - self.ct_min) / (self.ct_max - self.ct_min)
        return img * 2.0 - 1.0  # [-1, 1]

    # -------------------------
    # Spatial ops
    # -------------------------
    def _pad_to_target(self, img, target, padding):
        _, h, w = img.shape

        pad_h = max(target - h, 0)
        pad_w = max(target - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(
            img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=padding,
        )

    def _resize_to_target(self, img, target):
        return F.interpolate(
            img.unsqueeze(0),
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _standardize_spatial(self, img, modality):
        _, h, w = img.shape

        if h > self.target_size or w > self.target_size:
            img = self._resize_to_target(img, self.target_size)

        if modality == "CBCT":
            if h < self.target_size or w < self.target_size:
                img = self._pad_to_target(img, self.target_size, img.min())
        elif modality == "CT":
            if h < self.target_size or w < self.target_size:
                img = self._pad_to_target(img, self.target_size, -1000)
        return img

    def _resize(self, img, size):
        return F.interpolate(
            img.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    # -------------------------
    # Get item
    # -------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        modality = row["modality"]

        # Load image
        img = np.load(img_path).astype(np.float32)

        # Ensure (C, H, W)
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img[None, ...]

        img = torch.from_numpy(img)
        if modality == "CBCT":
            # Spatial standardization
            img = self._standardize_spatial(img, modality)
        elif modality == "CT":
            img = self._standardize_spatial(img, modality)

        # Normalize
        if modality == "CBCT":
            img = self._normalize_ct(img)
        elif modality == "CT":
            img = self._normalize_ct(img)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # 🔽 Final resize to 256x256 (just before return)
        img = self._resize(img, self.output_size)

        return {
            "img": img,
            "modality": modality,
            "path": img_path,
        }