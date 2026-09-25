"""BraTS: a one-off volume cache built from the NIfTI release, and a 3D dataset over it.

prepare_brats() stores each subject's co-registered modalities as one int16 array cropped
to the brain, together with the per-volume statistics used to normalize it. The dataset
memory-maps that array, so a training patch reads only its own voxels and no volume is
decompressed during training.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")
LABEL = "seg"
PERCENTILES = (0.5, 1.0, 2.0, 98.0, 99.0, 99.5)
CACHE_VERSION = 1


def read_split_file(path):
    """Return the split file's contents and its nonempty train/val/test subject lists."""
    values = json.loads(Path(path).read_text())
    splits = {name: values[name] for name in SPLITS if values.get(name)}
    if not splits:
        raise ValueError(f"{path} lists no subjects under {list(SPLITS)}")
    return values, splits


def subject_id(entry):
    """Name of the folder holding every file of one split entry, e.g. BraTS-GLI-00009-000."""
    parents = {Path(path).parent.name for path in entry.values()}
    if len(parents) != 1 or not next(iter(parents)):
        raise ValueError(f"Expected all files of one subject in its own folder: {entry}")
    return parents.pop()


def _percentile_key(value):
    return f"{float(value):g}"


def _bounds(mask, axis):
    indices = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
    return [int(indices[0]), int(indices[-1]) + 1]


def _save(path, array):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _prepare_subject(subject, files, modalities, label, output):
    import nibabel as nib

    images = {name: nib.load(path) for name, path in files.items()}
    reference = images[modalities[0]]
    for name, image in images.items():
        if image.ndim != 3 or image.shape != reference.shape or not np.allclose(
                image.affine, reference.affine, atol=1e-3):
            raise ValueError(f"{subject}: {name} is not on the voxel grid of {modalities[0]}")
    volume = np.stack([np.asanyarray(images[name].dataobj) for name in modalities])
    if not np.isfinite(volume).all():
        raise ValueError(f"{subject}: volumes contain NaN or infinity")
    brain = (volume != 0).any(0)
    if not brain.any():
        raise ValueError(f"{subject}: every modality is empty")
    bounds = [_bounds(brain, axis) for axis in range(3)]
    box = tuple(slice(start, stop) for start, stop in bounds)
    stats = {}
    for name, channel in zip(modalities, volume):
        values = channel[channel != 0].astype(np.float64)
        if not values.size:
            raise ValueError(f"{subject}: {name} is empty")
        levels = np.percentile(values, PERCENTILES)
        stats[name] = dict(percentiles={_percentile_key(p): float(v)
                                        for p, v in zip(PERCENTILES, levels)},
                           mean=float(values.mean()),
                           std=float(values.std()))
    int16 = np.iinfo(np.int16)
    compact = (np.issubdtype(volume.dtype, np.integer) and volume.min() >= int16.min
               and volume.max() <= int16.max)
    _save(output / f"{subject}.npy",
          np.ascontiguousarray(volume[(slice(None), *box)],
                               dtype=np.int16 if compact else np.float32))
    if label:
        segmentation = np.asanyarray(images[label].dataobj)
        if (segmentation.min() < 0 or segmentation.max() > 255
                or not np.array_equal(segmentation, np.round(segmentation))):
            raise ValueError(f"{subject}: {label} must hold integer labels in [0, 255]")
        _save(output / f"{subject}_{label}.npy",
              np.ascontiguousarray(segmentation[box], dtype=np.uint8))
    metadata = dict(shape=list(reference.shape),
                    affine=reference.affine.tolist(),
                    bounds=bounds,
                    stats=stats)
    # Written last: its presence marks the subject as complete when a run is resumed.
    temporary = output / f"{subject}.json.tmp"
    temporary.write_text(json.dumps(metadata))
    os.replace(temporary, output / f"{subject}.json")
    return subject


def prepare_brats(split_file, output_dir, *, data_dir=None, workers=1):
    """Cache every subject of a split file; already cached subjects are skipped.

    Files are resolved against data_dir, else the split file's "data_dir", else the split
    file's folder. The cache does not depend on the fold: one cache serves every split
    file that lists the same subjects and modalities.
    """
    values, splits = read_split_file(split_file)
    root = Path(data_dir or values.get("data_dir") or Path(split_file).parent)
    entries = {}
    for items in splits.values():
        for entry in items:
            if entries.setdefault(subject_id(entry), entry) != entry:
                raise ValueError(f"Subject {subject_id(entry)} is listed with different files")
    names = list(next(iter(entries.values())))
    if any(list(entry) != names for entry in entries.values()):
        raise ValueError(f"Every subject must list the same files: {names}")
    modalities = [name for name in names if name != LABEL]
    label = LABEL if LABEL in names else None
    output = Path(output_dir)
    volumes = output / "volumes"
    volumes.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    header = dict(version=CACHE_VERSION,
                  modalities=modalities,
                  label=label,
                  percentiles=[float(p) for p in PERCENTILES],
                  axes=["channel", "x", "y", "z"])
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if {key: previous.get(key) for key in header} != header:
            raise ValueError(f"{index_path} was built with different settings; use a new output_dir")
    jobs = [(name, {key: str(root / path) for key, path in entry.items()}, modalities, label,
             volumes) for name, entry in entries.items()
            if not (volumes / f"{name}.json").exists()]
    print(f"{len(entries)} subjects, {len(entries) - len(jobs)} already cached, "
          f"{len(jobs)} to prepare")
    failures, done = {}, 0

    def finished(name, error=None):
        nonlocal done
        done += 1
        if error is not None:
            failures[name] = repr(error)
        if done % 25 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} subjects")

    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(_prepare_subject, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                finished(futures[future], future.exception())
    else:
        for job in jobs:
            try:
                _prepare_subject(*job)
            except Exception as error:
                finished(job[0], error)
            else:
                finished(job[0])
    subjects = {path.stem: json.loads(path.read_text()) for path in sorted(volumes.glob("*.json"))}
    temporary = output / "index.json.tmp"
    temporary.write_text(json.dumps(dict(header, subjects=subjects)))
    os.replace(temporary, index_path)
    if failures:
        details = "\n".join(f"  {name}: {error}" for name, error in sorted(failures.items()))
        raise RuntimeError(f"{len(failures)} subjects failed and were not cached:\n{details}")
    return index_path


class BraTSVolumeDataset(Dataset):
    """3D volumes of one split, read from a prepare_brats() cache.

    Items follow the PairedDataset contract with one more spatial axis: condition and
    target are float32 C,X,Y,Z tensors in the requested modality order, and case_id is the
    subject name. Each modality of each volume is clipped to the given percentiles of its
    brain voxels and scaled to [-1, 1]; the background is -1.

    With patch_size, the train split yields samples_per_volume random cubic patches per
    subject and epoch, centred on a uniformly drawn voxel of the brain's bounding box;
    parts of a patch outside the volume are background. Every other split yields whole
    volumes, for sliding-window inference. An empty target list yields unlabeled items.

    Whole volumes also carry their affine, for NIfTI export, and evaluation regions:
    regions maps each name to null, for the brain mask, or to a list of segmentation
    labels (default: brain only; {} for none). max_subjects keeps the split's first
    subjects, e.g. to evaluate a costly configuration on a subset.
    """

    def __init__(self,
                 cache_dir,
                 split_file,
                 split,
                 condition,
                 target=(),
                 *,
                 patch_size=None,
                 samples_per_volume=1,
                 percentiles=(0.5, 99.5),
                 regions=None,
                 max_subjects=None):
        cache = Path(cache_dir)
        index_path = cache / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found; build the cache with scripts/prepare_brats.py")
        index = json.loads(index_path.read_text())
        _, splits = read_split_file(split_file)
        if split not in splits:
            raise ValueError(f"Split {split!r} is missing or empty in {split_file}")
        modalities = index["modalities"]
        condition, target = list(condition), list(target or [])
        if not condition or set(condition + target) - set(modalities):
            raise ValueError(f"condition and target must name modalities from {modalities}")
        if set(condition) & set(target):
            raise ValueError("condition and target modalities must not overlap")
        if len(percentiles) != 2 or not percentiles[0] < percentiles[1]:
            raise ValueError("percentiles must be an increasing [low, high] pair")
        low, high = (_percentile_key(p) for p in percentiles)
        if {low, high} - {_percentile_key(p) for p in index["percentiles"]}:
            raise ValueError(f"percentiles must be chosen from {index['percentiles']}")
        if patch_size is not None and patch_size < 1 or samples_per_volume < 1:
            raise ValueError("patch_size must be positive or null; samples_per_volume positive")
        if max_subjects is not None and max_subjects < 1:
            raise ValueError("max_subjects must be positive or null")
        self.regions = {"brain": None} if regions is None else {
            name: None if labels is None else [int(label) for label in labels]
            for name, labels in dict(regions).items()
        }
        if not index["label"] and any(labels is not None for labels in self.regions.values()):
            raise ValueError("Label regions need a cache built with segmentations")
        self.channels = [modalities.index(name) for name in condition + target]
        self.condition_channels = len(condition)
        self.patch_size = patch_size if split == "train" else None
        self.samples = samples_per_volume if self.patch_size is not None else 1
        self.subjects = []
        for entry in splits[split][:max_subjects]:
            name = subject_id(entry)
            meta = index["subjects"].get(name)
            if meta is None:
                raise ValueError(
                    f"{name} is not in {index_path}; rerun prepare_brats with {split_file}")
            levels = [[meta["stats"][m]["percentiles"][key] for m in modalities]
                      for key in (low, high)]
            self.subjects.append(
                dict(name=name,
                     path=str(cache / "volumes" / f"{name}.npy"),
                     labels=str(cache / "volumes" / f"{name}_{index['label']}.npy"),
                     affine=meta["affine"],
                     shape=meta["shape"],
                     bounds=meta["bounds"],
                     low=np.asarray(levels[0], np.float32)[self.channels, None, None, None],
                     high=np.asarray(levels[1], np.float32)[self.channels, None, None, None]))

    def __len__(self):
        return len(self.subjects) * self.samples

    def read(self, subject, origin, size):
        """Normalized channels of the box [origin, origin + size) of the original volume,
        and the box's brain mask."""
        image = np.full((len(self.channels), *size), -1, np.float32)
        mask = np.zeros(size, bool)
        source, destination = [], []
        for start, extent, (low, high) in zip(origin, size, subject["bounds"]):
            first, last = max(start, low), min(start + extent, high)
            if first >= last:
                return image, mask  # The box misses the brain entirely.
            source.append(slice(first - low, last - low))
            destination.append(slice(first - start, last - start))
        stored = np.load(subject["path"], mmap_mode="r")
        values = np.asarray(stored[(self.channels, *source)], dtype=np.float32)
        brain = (values != 0).any(0)
        scale = np.maximum(subject["high"] - subject["low"], 1e-6)
        values = np.clip((values - subject["low"]) / scale, 0, 1) * 2 - 1
        values[:, ~brain] = -1
        image[(slice(None), *destination)] = values
        mask[tuple(destination)] = brain
        return image, mask

    def __getitem__(self, index):
        subject = self.subjects[index // self.samples]
        shape = subject["shape"]
        if self.patch_size is None:
            image, brain = self.read(subject, (0, 0, 0), shape)
        else:
            size = (self.patch_size, ) * 3
            center = [np.random.randint(low, high) for low, high in subject["bounds"]]
            origin = [int(np.clip(c - s // 2, 0, max(n - s, 0)))
                      for c, s, n in zip(center, size, shape)]
            image, _ = self.read(subject, origin, size)
        split = self.condition_channels
        item = dict(condition=torch.from_numpy(np.ascontiguousarray(image[:split])),
                    case_id=subject["name"])
        if split < len(self.channels):
            item["target"] = torch.from_numpy(np.ascontiguousarray(image[split:]))
        if self.patch_size is None:
            item["affine"] = torch.tensor(subject["affine"], dtype=torch.float64)
            if self.regions:
                segmentation = None
                if any(labels is not None for labels in self.regions.values()):
                    segmentation = self.segmentation(subject)
                item["regions"] = {
                    name: torch.from_numpy(
                        brain if labels is None else np.isin(segmentation, labels))[None]
                    for name, labels in self.regions.items()
                }
        return item

    def segmentation(self, subject):
        """The subject's segmentation labels on the original volume grid."""
        labels = np.zeros(subject["shape"], np.uint8)
        labels[tuple(slice(low, high) for low, high in subject["bounds"])] = np.load(
            subject["labels"])
        return labels
