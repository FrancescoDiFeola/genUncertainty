"""Deterministic, nontrivial CPU models shared by legacy fixture generation and replay."""
import numpy as np
import torch
from torch import nn

from latent_uq.config import Config, Component
from latent_uq.models import Models


class ReferenceBackbone(nn.Module):

    def __init__(self, channels, uncertain):
        super().__init__()
        self.channels, self.uncertain = channels, uncertain
        self.gain = nn.Parameter(torch.tensor(0.13))
        self.context_gain = nn.Parameter(torch.tensor(0.05))
        self.logvar_offset = nn.Parameter(torch.tensor(-0.2))
        self.calls = []

    def forward(self, x, timesteps, context=None):
        self.calls.append((x.detach().clone(), timesteps.detach().clone(),
                           None if context is None else context.detach().clone()))
        channels = self.channels
        mean = self.gain * x[:, :channels] + 0.07 * x[:, channels:]
        mean = mean + 0.03 * timesteps.float().reshape(-1, 1, 1, 1) / 1000
        if context is not None:
            mean = mean + self.context_gain * context.mean((1, 2)).reshape(-1, 1, 1, 1)
        logvar = self.logvar_offset + 0.15 * mean + torch.arange(channels).reshape(
            1, channels, 1, 1) * 0.1
        return (mean, logvar) if self.uncertain else mean


class ReferenceContext(nn.Module):

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return (self.bias + self.gain * x.mean((1, 2, 3))).reshape(-1, 1, 1).expand(-1, 1, 128)


class ReferenceVAE(nn.Module):

    def encode(self, x):
        latent = torch.cat([x, x * 0.5 + 0.1, x * 0.25 - 0.1], 1)
        return latent, torch.ones_like(latent)

    def decode(self, z):
        return z[:, :1] + 0.2 * z[:, 1:2].square() + 0.3 * z[:, 2:3]

    def forward(self, x):
        mean, sigma = self.encode(x)
        return self.decode(mean + torch.randn_like(mean) * sigma), mean, sigma


def seed():
    torch.manual_seed(29)
    np.random.seed(29)


def images():
    condition = torch.linspace(-0.7, 0.8, 64).reshape(1, 1, 8, 8)
    target = condition.flip(-1) * 0.8 + torch.linspace(0, 0.3, 8).reshape(1, 1, 8, 1)
    target[0, 0, 0, 0] = 0  # Exercise the vanilla foreground mask.
    target[0, 0, 0, 1] = -1  # Exercise the summary foreground mask.
    return condition, target


def setup(framework,
          mode,
          *,
          analysis='metrics',
          posthoc=False,
          ablation=False,
          calibration=False,
          concat=False):
    config = Config(framework=framework, mode=mode)
    config.data.batch_size = 1
    config.inference.analyses = [analysis]
    config.inference.self_conditioning = not ablation
    if concat:
        config.model.context_input = "prediction_variance"
    if posthoc:
        config.inference.uncertainty = 'posthoc'
    channels = 3 if config.latent else 1
    if config.latent:
        config.model.autoencoder = Component('tests.reference_cases.ReferenceVAE',
                                             checkpoint='fixture')
        config.model.scaling_factor = 1.7
    if calibration:
        config.model.uncertainty_decoder = Component('unused.fixture')
    backbone = ReferenceBackbone(channels, mode != 'base')
    context = ReferenceContext() if mode == 'selfcond' else None
    decoder = None
    if calibration:
        decoder = nn.Conv2d(3, 1, 1)
        with torch.no_grad():
            decoder.weight.copy_(torch.tensor([0.2, -0.3, 0.4]).reshape_as(decoder.weight))
            decoder.bias.fill_(0.1)
    models = Models(backbone, context, ReferenceVAE() if config.latent else None, decoder)
    return config, models
