from __future__ import annotations

from typing import Any, Mapping


def get_condition(batch: Mapping[str, Any]) -> Any:
    """Return the conditioning/source tensor from a batch."""
    if "condition" in batch:
        return batch["condition"]
    if "A" in batch:
        return batch["A"]
    raise KeyError(
        "Dataset batch must contain 'condition'. "
        f"Available keys: {list(batch.keys())}"
    )


def get_target(batch: Mapping[str, Any]) -> Any:
    """Return the target tensor from a batch."""
    if "target" in batch:
        return batch["target"]
    if "B" in batch:
        return batch["B"]
    raise KeyError(
        "Dataset batch must contain 'target'. "
        f"Available keys: {list(batch.keys())}"
    )


def get_case_id(batch: Mapping[str, Any], default: Any = None) -> Any:
    return batch.get("case_id", default)


def get_condition_target_case_id(batch: Mapping[str, Any], default_case_id: Any = None):
    """Return condition, target and case_id from a standardized or legacy batch."""
    return get_condition(batch), get_target(batch), get_case_id(batch, default_case_id)
