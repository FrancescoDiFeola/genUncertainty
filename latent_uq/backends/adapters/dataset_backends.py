from __future__ import annotations

from typing import Any, Tuple


def build_dataset_backend(args: Any) -> Tuple[Any, float]:
    """Deprecated dataset backend.

    Dataset selection by task name has been removed from the public API because
    it makes the framework difficult to extend. Use ``data.dataset_class`` and
    ``data.dataset_kwargs`` instead.
    """
    raise RuntimeError(
        "Task-specific dataset backends are disabled. Add a dataset by setting "
        "data.dataset_class to a fully-qualified class path and passing any "
        "constructor arguments through data.dataset_kwargs."
    )
