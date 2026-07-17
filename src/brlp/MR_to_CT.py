"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict

def build_pairs_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    pairs = defaultdict(dict)
    mr_paths_by_subject = defaultdict(list)

    for _, row in df.iterrows():

        img_name = row["img_name"]
        img_path = row["img_path"]
        modality = row["modality"].upper()
        parts = img_name.split("_")
        subject_id = parts[0]
        slice_id = parts[-1]
        key = (subject_id, slice_id)
        pairs[key][modality] = img_path
        if modality == "MR":
            mr_paths_by_subject[subject_id].append(img_path)

    paired_samples = []

    for (subject_id, slice_id), entry in pairs.items():
        if "MR" in entry and "CT" in entry:
            paired_samples.append({
                "subject_id": subject_id,
                "slice_id": slice_id,
                "mr_path": entry["MR"],
                "ct_path": entry["CT"],
            })
    paired_samples = sorted(
        paired_samples,
        key=lambda x: (x["subject_id"], int(x["slice_id"]))
    )

    return paired_samples, mr_paths_by_subject

class MRCTPaired(Dataset):

    def __init__(
        self,
        csv_path: str,
        target_size: int = 512,
        output_size: int = 256,
        ct_min: float = -1000.0,
        ct_max: float = 2000.0,
        mr_percentiles=(1, 99),
    ):

        self.samples, self.mr_paths_by_subject = build_pairs_from_csv(csv_path)
        self.target_size = target_size
        self.output_size = output_size
        self.ct_min = ct_min
        self.ct_max = ct_max
        self.mr_percentiles = mr_percentiles

        self.mr_norm_params = self._compute_mr_norm_params()

    def __len__(self):
        return len(self.samples)

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
            )

    def _resize_to_target(self, img, size):
        return F.interpolate(
            img.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _compute_mr_norm_params(self):
        params = {}
        for subject_id, paths in self.mr_paths_by_subject.items():
            values = []
            for path in paths:
                img = np.load(path).astype(np.float32)
                # exclude pure background if present
                valid = img[img != 0]
                if valid.size > 0:
                    values.append(valid.reshape(-1))
                else:
                    values.append(img.reshape(-1))
            values = np.concatenate(values)
            p_low, p_high = np.percentile(values, self.mr_percentiles)
            if p_high <= p_low:
                p_low = float(values.min())
                p_high = float(values.max() + 1e-8)
            params[subject_id] = (float(p_low), float(p_high))

        return params

    def _normalize_mr(self, img, subject_id):
        p_low, p_high = self.mr_norm_params[subject_id]
        img = torch.clamp(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8)

        return img * 2.0 - 1.0

    def _normalize_ct(self, img):
        img = torch.clamp(img, self.ct_min, self.ct_max)
        img = (img - self.ct_min) / (self.ct_max - self.ct_min)
        return img * 2.0 - 1.0  # [-1, 1]

    def _standardize_spatial(self, img, modality):
        _, h, w = img.shape
        if h > self.target_size or w > self.target_size:
            img = self._resize_to_target(img, self.target_size)
        _, h, w = img.shape
        if h < self.target_size or w < self.target_size:
            pad_value = -1.0 if modality == "CT" else img.min().item()
            img = self._pad_to_target(img, pad_value)

        return img

    # ------------------------- #
    # 		  Get item          #
    # ------------------------- #
    def __getitem__(self, idx):
        sample = self.samples[idx]

        mr = np.load(sample["mr_path"]).astype(np.float32)
        ct = np.load(sample["ct_path"]).astype(np.float32)

        if mr.ndim == 2:
            mr = mr[None, ...]
        if ct.ndim == 2:
            ct = ct[None, ...]

        mr = torch.from_numpy(mr)
        ct = torch.from_numpy(ct)

        # Normalize
        mr = self._normalize_mr(mr, sample["subject_id"])
        ct = self._normalize_ct(ct)

        # Shared spatial processing
        mr = self._standardize_spatial(mr, modality="MR")
        ct = self._standardize_spatial(ct, modality="CT")

        # Final resize
        mr = self._resize_to_target(mr, self.output_size)
        ct = self._resize_to_target(ct, self.output_size)

        return {
            "A": mr,   # MRI
            "B": ct,   # CT
        }
"""
import numpy as np

import torch

import torch.nn.functional as F

import pandas as pd

from torch.utils.data import Dataset

from collections import defaultdict


def build_pairs_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    pairs = defaultdict(dict)

    mr_paths_by_subject = defaultdict(list)

    for _, row in df.iterrows():

        img_name = row["img_name"]

        img_path = row["img_path"]

        modality = row["modality"].upper()

        subject_id = str(row["subject_id"]) if "subject_id" in row else img_name.split("_")[0]

        slice_id = str(row["slice_id"]) if "slice_id" in row else img_name.split("_")[-1]

        key = (subject_id, slice_id)

        pairs[key][modality] = {

            "path": img_path,

            "spacing_x": float(row["spacing_x"]),

            "spacing_y": float(row["spacing_y"]),

        }

        if modality == "MR":
            mr_paths_by_subject[subject_id].append(img_path)

    paired_samples = []

    for (subject_id, slice_id), entry in pairs.items():

        if "MR" in entry and "CT" in entry:
            paired_samples.append({

                "subject_id": subject_id,

                "slice_id": slice_id,

                "mr_path": entry["MR"]["path"],

                "ct_path": entry["CT"]["path"],

                "mr_spacing": (

                    entry["MR"]["spacing_x"],

                    entry["MR"]["spacing_y"],

                ),

                "ct_spacing": (

                    entry["CT"]["spacing_x"],

                    entry["CT"]["spacing_y"],

                ),

            })

    paired_samples = sorted(

        paired_samples,

        key=lambda x: (x["subject_id"], int(x["slice_id"]))

    )

    return paired_samples, mr_paths_by_subject


class MRCTPaired(Dataset):

    def __init__(
            self,
            csv_path: str,
            output_size: int = 256,
            target_spacing=(1.5, 1.5),
            ct_min: float = -1000.0,
            ct_max: float = 2000.0,
            mr_percentiles=(1, 99),
            crop_if_needed: bool = True,
    ):

        self.samples, self.mr_paths_by_subject = build_pairs_from_csv(csv_path)
        self.target_size = output_size
        self.target_spacing = target_spacing
        self.ct_min = ct_min
        self.ct_max = ct_max
        self.mr_percentiles = mr_percentiles
        self.crop_if_needed = crop_if_needed
        self.mr_norm_params = self._compute_mr_norm_params()

    def __len__(self):
        return len(self.samples)

    def _compute_mr_norm_params(self):
        params = {}
        for subject_id, paths in self.mr_paths_by_subject.items():
            values = []
            for path in paths:
                img = np.load(path).astype(np.float32)
                valid = img[img != 0]
                if valid.size > 0:
                    values.append(valid.reshape(-1))
                else:
                    values.append(img.reshape(-1))
            values = np.concatenate(values)
            p_low, p_high = np.percentile(values, self.mr_percentiles)

            if p_high <= p_low:
                p_low = float(values.min())
                p_high = float(values.max() + 1e-8)
            params[subject_id] = (float(p_low), float(p_high))

        return params

    def _normalize_mr(self, img, subject_id):
        p_low, p_high = self.mr_norm_params[subject_id]
        img = torch.clamp(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8)
        return img * 2.0 - 1.0

    def _normalize_ct(self, img):
        img = torch.clamp(img, self.ct_min, self.ct_max)
        img = (img - self.ct_min) / (self.ct_max - self.ct_min)
        return img * 2.0 - 1.0

    def _resample_to_spacing(self, img, original_spacing, mode="bilinear"):
        """
        img shape: (C, H, W)
        original_spacing: (spacing_x, spacing_y)
        target_spacing:   (target_spacing_x, target_spacing_y)
        Note:
        H corresponds to y-axis, W corresponds to x-axis.
        """

        spacing_x, spacing_y = original_spacing
        target_spacing_x, target_spacing_y = self.target_spacing
        _, h, w = img.shape
        new_h = int(round(h * spacing_y / target_spacing_y))
        new_w = int(round(w * spacing_x / target_spacing_x))
        img = F.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode=mode,
            align_corners=False if mode in ["bilinear", "bicubic"] else None,
        ).squeeze(0)
        return img

    def _center_crop_if_needed(self, img):
        _, h, w = img.shape
        if h <= self.target_size and w <= self.target_size:
            return img

        if not self.crop_if_needed:
            raise ValueError(
                f"Image size after resampling is {(h, w)}, "
                f"larger than target_size={self.target_size}. "
                "Increase target_size or enable crop_if_needed."
            )

        top = max((h - self.target_size) // 2, 0)
        left = max((w - self.target_size) // 2, 0)
        return img[
               :,
               top:top + min(h, self.target_size),
               left:left + min(w, self.target_size),
               ]

    def _pad_to_target(self, img, pad_value=-1.0):
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
            value=float(pad_value),
        )

    def _standardize_spatial(self, img, spacing, modality):
        if modality == "MR":
            interp_mode = "bilinear"
        elif modality == "CT":
            interp_mode = "bilinear"
        else:
            interp_mode = "bilinear"
        img = self._resample_to_spacing(
            img,
            original_spacing=spacing,
            mode=interp_mode,
        )

        img = self._center_crop_if_needed(img)
        # Dopo normalizzazione, lo sfondo MR e CT è coerentemente circa -1.
        img = self._pad_to_target(
            img,
            pad_value=-1.0,
        )

        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mr = np.load(sample["mr_path"]).astype(np.float32)
        ct = np.load(sample["ct_path"]).astype(np.float32)
        if mr.ndim == 2:
            mr = mr[None, ...]

        if ct.ndim == 2:
            ct = ct[None, ...]

        mr = torch.from_numpy(mr)
        ct = torch.from_numpy(ct)
        mr = self._normalize_mr(mr, sample["subject_id"])
        ct = self._normalize_ct(ct)
        mr = self._standardize_spatial(
            mr,
            spacing=sample["mr_spacing"],
            modality="MR",
        )

        ct = self._standardize_spatial(
            ct,
            spacing=sample["ct_spacing"],
            modality="CT",
        )

        return {
            "A": mr,
            "B": ct,
            "condition": mr,
            "target": ct,
            "case_id": f"{sample['subject_id']}_{sample['slice_id']}",
        }
