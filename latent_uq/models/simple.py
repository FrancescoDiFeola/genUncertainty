from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvBackbone(nn.Module):
    """Small task-agnostic backbone for smoke tests and examples.

    It accepts the same call styles used by the framework and can return either a
    single prediction or prediction plus log-variance.
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1, hidden_channels: int = 16, aleatoric: bool = False):
        super().__init__()
        self.aleatoric = bool(aleatoric)
        n_out = out_channels * (2 if self.aleatoric else 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, n_out, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps=None, context=None):
        y = self.net(x)
        if not self.aleatoric:
            return {"prediction": y}
        pred, logvar = torch.chunk(y, 2, dim=1)
        return {"prediction": pred, "logvar": logvar}
