from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.utils.data import Dataset


class BasePairedDataset(Dataset, ABC):
    """Task-agnostic paired dataset interface used by Latent-UQ.

    Each item must return a dictionary with at least:

        {
            "condition": source_or_conditioning_tensor,
            "target": target_tensor,
            "case_id": optional_identifier,
        }

    The public training/inference entrypoints only depend on this contract and
    do not need to know the clinical task, modality pair, scanner, or dataset
    layout. Dataset-specific logic must stay inside the user-provided dataset
    class.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError
