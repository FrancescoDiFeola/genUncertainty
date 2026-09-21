import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "genunc-mpl"))

import torch
import pytest
from scripts.make_example import make_example
from latent_uq.config import Config, Component
from tests.helpers import VAE


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def config_factory(tmp_path):
    csv_path = make_example(tmp_path / "data", count=3, size=8)
    vae_path = tmp_path / "vae.pt"
    torch.save(VAE().state_dict(), vae_path)

    def make(framework="dm", mode="selfcond"):
        config = Config(framework=framework, mode=mode)
        config.data.kwargs = dict(csv_path=str(csv_path))
        config.model.context_dim = 4
        channels = 2 if config.latent else 1
        config.model.backbone = Component(
            "tests.helpers.Backbone",
            dict(channels=channels,
                 condition_channels=channels,
                 uncertainty=mode != "base",
                 context_dim=4))
        config.model.context_encoder = Component("tests.helpers.Context",
                                                 dict(channels=channels, context_dim=4))
        if config.latent:
            config.model.latent_channels = 2
            config.model.autoencoder = Component("tests.helpers.VAE", checkpoint=str(vae_path))
            config.model.scaling_factor = 2.5
        config.inference.steps = 4
        config.inference.last_k = 2
        config.inference.decode_samples = 8
        config.inference.samples = 3
        config.validate()
        return config

    return make
