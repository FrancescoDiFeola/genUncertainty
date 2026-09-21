"""One inference module per model family: dm, fm, ldm, lfm."""
from importlib import import_module

MODULES = dict(dm='inference_ddpm', fm='inference_RF', ldm='inference_LDM', lfm='inference_LFM')


def module_for(framework):
    return import_module(f'{__name__}.{MODULES[framework]}')
