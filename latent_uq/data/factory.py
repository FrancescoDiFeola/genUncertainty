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
    scaling_factor = float(
        getattr(args, "scaling_factor", 1.0) or 1.0
    )

    dataset_class = getattr(args, "dataset_class", None)

    # Copia per evitare di modificare direttamente il dizionario YAML.
    dataset_kwargs = dict(
        getattr(args, "dataset_kwargs", None) or {}
    )

    if not dataset_class:
        raise ValueError(
            "Missing data.dataset_class. Provide a built-in alias or "
            "a fully-qualified dataset class."
        )

    # Compatibilità con la vecchia CLI e con le configurazioni legacy.
    legacy_argument_names = (
        "annotation_A",
        "annotation_B",
        "csv_path",
        "dataroot",
        "output_size",
        "motion_level",
    )

    for argument_name in legacy_argument_names:
        argument_value = getattr(args, argument_name, None)

        # I valori inseriti esplicitamente nella CLI/config legacy
        # completano dataset_kwargs senza sovrascrivere quelli già presenti.
        if (
            argument_value is not None
            and argument_name not in dataset_kwargs
        ):
            dataset_kwargs[argument_name] = argument_value

    condition_key = getattr(args, "condition_key", None)
    target_key = getattr(args, "target_key", None)

    cls = import_object(dataset_class)

    try:
        dataset = cls(**dataset_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Could not initialize dataset '{dataset_class}'. "
            f"Arguments passed to the dataset: "
            f"{sorted(dataset_kwargs.keys())}. "
            f"Original error: {exc}"
        ) from exc

    dataset = StandardizedPairedDataset(
        dataset,
        condition_key=condition_key,
        target_key=target_key,
    )

    return dataset, scaling_factor