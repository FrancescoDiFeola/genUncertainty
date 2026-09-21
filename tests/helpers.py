"""Controlled models for algorithm tests; not example datasets or production models."""
import torch
from torch import nn

from latent_uq.analysis import CustomAnalysis


class ThresholdedUncertainty(CustomAnalysis):
    """Example custom analysis: fraction of pixels whose uncertainty exceeds a threshold."""
    columns = ['fraction_above_threshold']
    requires_uncertainty = True

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def __call__(self, target, prediction, uncertainty):
        return [dict(fraction_above_threshold=float((uncertainty > self.threshold).mean()))]


class Backbone(nn.Module):

    def __init__(self, channels=1, condition_channels=1, uncertainty=True, context_dim=4):
        super().__init__()
        self.conv = nn.Conv2d(channels + condition_channels, channels, 1)
        self.time = nn.Linear(1, channels)
        self.context = nn.Linear(context_dim, channels)
        self.logvar = nn.Parameter(torch.linspace(-1, 0, channels).reshape(1, channels, 1, 1))
        self.uncertainty = uncertainty
        self.calls = []

    def forward(self, x, timesteps, context=None):
        self.calls.append(
            (x.detach().clone(), timesteps.detach().clone(),
             None if context is None else context.detach().clone(), torch.is_grad_enabled()))
        y = self.conv(x) + self.time(timesteps.float()[:, None] / 1000)[:, :, None, None]
        if context is not None:
            y = y + self.context(context.mean(1))[:, :, None, None]
        return (y, self.logvar.expand_as(y)) if self.uncertainty else y


class Context(nn.Module):

    def __init__(self, channels=1, context_dim=4):
        super().__init__()
        self.linear = nn.Linear(channels, context_dim)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.linear(x.mean((2, 3))).unsqueeze(1)


class VAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5))

    def encode(self, x):
        return torch.cat([x, x * self.scale], dim=1), torch.zeros_like(x)

    def decode(self, x):
        return x[:, :1] + 0.1 * x[:, 1:2].square()

    def forward(self, x):
        mean, _ = self.encode(x)
        sigma = torch.ones_like(mean)
        return self.decode(mean + torch.randn_like(mean) * sigma), mean, sigma
