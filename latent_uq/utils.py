"""Small shared utilities; no project-specific import side effects."""
from contextlib import contextmanager
import importlib
import random

import numpy as np
import torch


def import_object(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def evaluating(*modules):
    states = [(child, child.training) for module in modules if module is not None
              for child in module.modules()]
    try:
        for module in modules:
            if module is not None:
                module.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


def finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return tensor
