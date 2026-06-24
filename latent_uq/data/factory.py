from __future__ import annotations

import importlib
from typing import Any, Tuple

from latent_uq.data.adapters import StandardizedPairedDataset
from latent_uq.data.legacy import BUILTIN_DATASETS


def import_object(path: str):
    """Import an object from a fully-qualified Python path or built-in alias."""
    if not path or not isinstance(path, str):
        raise ValueError("A non-empty dataset_class path or built-in dataset alias is required.")
    resolved = BUILTIN_DATASETS.get(path.lower(), path)
    if "." not in resolved:
        raise ValueError(
            f"Unknown dataset alias '{path}'. Use one of {sorted(BUILTIN_DATASETS)} "
            "or a fully-qualified class path such as my_project.datasets.MyDataset."
        )
    module_name, object_name = resolved.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_dataset(args: Any) -> Tuple[Any, float]:
    """Build a dataset from ``data.dataset_class`` and ``data.dataset_kwargs``.

    This function is intentionally task-agnostic. To add a new dataset, define a
    class that follows :class:`latent_uq.data.base.BasePairedDataset` and point
    the YAML configuration to it, for example:

        data:
          dataset_class: my_project.datasets.MyDataset
          dataset_kwargs:
            root: /path/to/data
            split: test
          scaling_factor: 1.0

    No changes to Latent-UQ source code are required.
    """
    scaling_factor = float(getattr(args, "scaling_factor", 1.0) or 1.0)
    dataset_class = getattr(args, "dataset_class", None)
    dataset_kwargs = getattr(args, "dataset_kwargs", None) or {}

    if not dataset_class:
        raise ValueError(
            "Missing data.dataset_class. Latent-UQ is task-agnostic and no longer "
            "selects datasets from task names. Provide, for example:\n\n"
            "data:\n"
            "  dataset_class: my_project.datasets.MyDataset\n"
            "  dataset_kwargs:\n"
            "    root: /path/to/data\n"
            "    split: test\n"
            "  scaling_factor: 1.0\n"
        )

    condition_key = getattr(args, "condition_key", None)
    target_key = getattr(args, "target_key", None)

    cls = import_object(dataset_class)
    dataset = cls(**dataset_kwargs)
    dataset = StandardizedPairedDataset(dataset, condition_key=condition_key, target_key=target_key)
    return dataset, scaling_factor
