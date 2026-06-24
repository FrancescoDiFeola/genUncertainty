from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    logvar: Optional[torch.Tensor] = None
    extra: Optional[Any] = None

    @property
    def has_uncertainty(self) -> bool:
        return self.logvar is not None
