"""Minimal custom dataset example for Latent-UQ.

Copy this file into your own project and replace the placeholder tensor loading
logic with your image loading code.
"""
from __future__ import annotations

import pandas as pd
import torch

from latent_uq.data.base import BasePairedDataset


class MyPairedDataset(BasePairedDataset):
    """Example paired dataset.

    The dataset must return a dictionary containing at least:

        condition: conditioning/source tensor
        target: target tensor
        case_id: sample identifier
    """

    def __init__(self, csv_path: str, image_size: int = 256):
        self.table = pd.read_csv(csv_path)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]

        # Replace these placeholders with actual image loading.
        condition = torch.zeros(1, self.image_size, self.image_size)
        target = torch.zeros(1, self.image_size, self.image_size)

        return {
            "condition": condition,
            "target": target,
            "case_id": str(row.get("case_id", idx)),
        }
