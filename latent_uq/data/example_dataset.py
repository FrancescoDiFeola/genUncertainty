from __future__ import annotations

from typing import Dict

import torch

from latent_uq.data.base import BasePairedDataset


class RandomPairedDataset(BasePairedDataset):
    """Small synthetic paired dataset for smoke tests and examples.

    It returns random tensors with the standard task-agnostic contract:

        {"condition": source, "target": target, "case_id": id}
    """

    def __init__(self, length: int = 4, channels: int = 1, height: int = 256, width: int = 256):
        self.length = int(length)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        condition = torch.randn(self.channels, self.height, self.width)
        target = torch.randn(self.channels, self.height, self.width)
        return {"condition": condition, "target": target, "case_id": f"random_{index:04d}"}
