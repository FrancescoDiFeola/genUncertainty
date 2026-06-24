from __future__ import annotations

from typing import Any, Dict, Mapping

from torch.utils.data import Dataset


class StandardizedPairedDataset(Dataset):
    """Wrap an arbitrary paired dataset and expose Latent-UQ standard keys.

    The wrapped dataset may already return ``condition``/``target`` or may use
    the historical ``A``/``B`` keys. The returned item always contains:

        {"condition": ..., "target": ..., "case_id": ...}

    Extra keys from the original sample are preserved.
    """

    def __init__(self, dataset: Dataset, condition_key: str | None = None, target_key: str | None = None):
        self.dataset = dataset
        self.condition_key = condition_key
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.dataset[index]
        if not isinstance(item, Mapping):
            raise TypeError(
                "Latent-UQ paired datasets must return a mapping/dict. "
                f"Got {type(item)!r} from {self.dataset.__class__.__name__}."
            )
        out = dict(item)

        condition_key = self.condition_key
        target_key = self.target_key
        if condition_key is None:
            condition_key = "condition" if "condition" in out else "A"
        if target_key is None:
            target_key = "target" if "target" in out else "B"

        if condition_key not in out:
            raise KeyError(
                f"Condition key '{condition_key}' not found. Available keys: {list(out.keys())}. "
                "Use data.condition_key in the YAML if your dataset uses a custom key."
            )
        if target_key not in out:
            raise KeyError(
                f"Target key '{target_key}' not found. Available keys: {list(out.keys())}. "
                "Use data.target_key in the YAML if your dataset uses a custom key."
            )

        out["condition"] = out[condition_key]
        out["target"] = out[target_key]
        out.setdefault("case_id", out.get("A_paths", out.get("B_paths", str(index))))
        return out
