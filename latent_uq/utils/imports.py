from __future__ import annotations

import importlib
from typing import Any


def import_object(path: str) -> Any:
    """Import an object from a fully-qualified Python path.

    Examples
    --------
    >>> import_object("torch.nn.MSELoss")
    <class 'torch.nn.modules.loss.MSELoss'>
    """
    if not path or not isinstance(path, str):
        raise ValueError("Expected a non-empty fully-qualified object path.")
    if "." not in path:
        raise ValueError(f"Invalid object path '{path}'. Use e.g. package.module.ClassName")
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"Object '{object_name}' not found in module '{module_name}'.") from exc
