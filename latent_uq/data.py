"""One example dataset: CSV-indexed NumPy image pairs with explicit preprocessing."""
from pathlib import Path
import csv

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .utils import finite, import_object, seed_worker


class PairedDataset(Dataset):
    """CSV columns: condition, optional target, optional case_id.

    Paths are relative to the CSV directory. Arrays must already be normalized,
    floating-point H,W or C,H,W images. No resizing or intensity scaling is hidden
    in this loader. All rows must either have targets or omit them.
    """

    def __init__(self, csv_path: str):
        path = Path(csv_path)
        self.root = path.resolve().parent
        with path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows or any(not row.get("condition") for row in self.rows):
            raise ValueError("Dataset CSV must contain at least one condition path")
        targets = [bool(row.get("target")) for row in self.rows]
        if any(targets) and not all(targets):
            raise ValueError("Either all rows must have targets or all must omit them")
        ids = [row.get("case_id") or str(i) for i, row in enumerate(self.rows)]
        if len(set(ids)) != len(ids):
            raise ValueError("case_id values must be unique")
        self.ids = ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = {"case_id": self.ids[index]}
        for key in ("condition", "target"):
            name = self.rows[index].get(key)
            if not name:
                continue
            array = np.load(self.root / name, allow_pickle=False)
            if not np.issubdtype(array.dtype, np.floating) or array.ndim not in (2, 3):
                raise ValueError(f"{name}: expected a floating-point H,W or C,H,W array")
            if array.ndim == 2:
                array = array[None]
            item[key] = finite(torch.from_numpy(array.copy()).float(), name)
        return item


def make_loader(config, *, training):
    dataset = import_object(config.data.class_path)(**config.data.kwargs)
    if len(dataset) < 1:
        raise ValueError("Dataset must not be empty")
    if training and config.data.drop_last and len(dataset) < config.data.batch_size:
        raise ValueError("drop_last=True would produce no training batches; reduce batch_size")
    workers = {}
    if config.data.num_workers > 0:
        workers = dict(persistent_workers=config.data.persistent_workers,
                       prefetch_factor=config.data.prefetch_factor)
    return DataLoader(dataset,
                      batch_size=config.data.batch_size,
                      shuffle=training,
                      num_workers=config.data.num_workers,
                      drop_last=training and config.data.drop_last,
                      worker_init_fn=seed_worker,
                      pin_memory=config.data.pin_memory,
                      **workers)


def prepare_batch(batch, config, *, require_target=False):
    if not isinstance(batch, dict) or "condition" not in batch:
        raise ValueError("Datasets must return a dictionary containing condition")
    condition = batch["condition"].to(config.device, dtype=torch.float32)
    target = batch.get("target")
    if require_target and target is None:
        raise ValueError("Training and error analyses require target images")
    if target is not None:
        target = target.to(config.device, dtype=torch.float32)
    axes = "H,W" if config.model.spatial_dims == 2 else "D,H,W"
    for name, value, channels in [("condition", condition, config.model.condition_channels),
                                  ("target", target, config.model.target_channels)]:
        if value is None:
            continue
        if (value.ndim != 2 + config.model.spatial_dims or value.shape[1] != channels
                or min(value.shape) < 1):
            raise ValueError(f"{name} must be N,{channels},{axes}; got {tuple(value.shape)}")
        finite(value, name)
    if target is not None and (condition.shape[0] != target.shape[0]
                               or condition.shape[2:] != target.shape[2:]):
        raise ValueError("Condition and target must have matching batch and spatial dimensions")
    return condition, target


def random_crop_pair(condition, target, size, pad_value=-1):
    """One shared random crop per pair batch, over every spatial axis (2D or 3D);
    mismatched images are never truncated."""
    if condition.shape[0] != target.shape[0] or condition.shape[2:] != target.shape[2:]:
        raise ValueError("Paired crops require matching batch and spatial dimensions")
    padding = []
    for extent in reversed(condition.shape[2:]):  # F.pad lists the last axis first.
        missing = max(size - extent, 0)
        padding += [missing // 2, missing - missing // 2]
    condition, target = [F.pad(x, padding, value=pad_value) for x in (condition, target)]
    starts = [int(torch.randint(extent - size + 1, (1, ))) for extent in condition.shape[2:]]
    window = (..., *[slice(start, start + size) for start in starts])
    return condition[window], target[window]
