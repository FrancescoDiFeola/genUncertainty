from __future__ import annotations

import torch
import torch.nn as nn


class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic aleatoric regression loss.

    precision = exp(-logvar)
    L = mean(0.5 * precision * (target - pred_mean)^2 + 0.5 * logvar)
        + reg_weight * mean(precision)
    """

    def __init__(self, min_logvar: float = -7.0, reg_weight: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.min_logvar = min_logvar
        self.reg_weight = reg_weight
        self.reduction = reduction

    def forward(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_logvar = torch.clamp(pred_logvar, min=self.min_logvar)
        precision = torch.exp(-pred_logvar)
        base_loss = 0.5 * precision * (target - pred_mean) ** 2 + 0.5 * pred_logvar
        reg = precision.mean()
        if self.reduction == "mean":
            loss = base_loss.mean()
        elif self.reduction == "sum":
            loss = base_loss.sum()
        else:
            loss = base_loss
        return loss + self.reg_weight * reg
