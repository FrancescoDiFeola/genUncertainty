"""PNG training monitor, a file-based alternative to TensorBoard.

Two images in <output_dir>/monitor/ are replaced in place while training runs, so an
image viewer or an editor tab always shows the latest state:

- losses.png, every training.plot_every epochs: one panel per quantity that
  training.jsonl records, i.e. the loss and, for aleatoric/selfcond, its components.
- samples.png, every training.sample_every epochs and after the first and the last
  epoch: fixed examples generated with the current weights, next to their inputs,
  target, error and, for aleatoric/selfcond, the propagated standard deviation.

Monitoring never changes training: its examples are drawn and generated under a fixed
seed, with every random state restored afterwards, and a failure is reported on stderr
while training continues.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

from .data import prepare_batch
from .sampling import sample
from .utils import import_object, seed_everything

# Chart chrome for a light surface, and the one series color.
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"
HEAT = "inferno"  # Error and standard deviation maps, each with its own color bar.
TITLES = dict(loss="Training loss",
              mse="MSE of the network output",
              logvar="Mean predicted log variance",
              calibration="Calibration loss (unweighted)")
SKIPPED = {"epoch", "seconds", "max_memory_gb"}  # Bookkeeping, not curves.


def require_matplotlib():
    """Fail when training starts, not after its first epoch, if matplotlib is missing."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise ImportError("PNG monitoring needs matplotlib: python -m pip install matplotlib, "
                          "or set training.plot_every and training.sample_every to 0") from error


@contextmanager
def preserved_random_state(device):
    """Run a block without moving any random sequence that training draws from."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def load_examples(config):
    """training.sample_count examples, evenly spaced over the dataset that data.kwargs
    updated with training.sample_data selects, kept on the CPU as batches of one."""
    kwargs = {**config.data.kwargs, **config.training.sample_data}
    with preserved_random_state(config.device):
        seed_everything(config.seed)  # The same random patches in every run and resume.
        dataset = import_object(config.data.class_path)(**kwargs)
        if len(dataset) < 1:
            raise ValueError("The monitoring dataset (training.sample_data) is empty")
        indices = np.linspace(0, len(dataset) - 1, config.training.sample_count).round()
        items = [(index, dataset[index]) for index in sorted({int(i) for i in indices})]
    cpu, examples = replace(config, device="cpu"), []
    for index, item in items:
        if not isinstance(item, dict) or item.get("target") is None:
            raise ValueError("Monitoring examples need targets: select a labeled dataset "
                             "with training.sample_data")
        batch = {key: torch.as_tensor(item[key])[None] for key in ("condition", "target")}
        condition, target = prepare_batch(batch, cpu, require_target=True)
        examples.append(
            dict(name=str(item.get("case_id", index)), condition=condition, target=target))
    return examples


class Monitor:
    """Writes <output_dir>/monitor/losses.png and samples.png after training epochs."""

    def __init__(self, config, output_dir):
        require_matplotlib()
        self.config, self.run = config, Path(output_dir)
        self.root = self.run / "monitor"
        # Loaded once, so that every epoch shows the same examples; a bad
        # training.sample_data fails here, before training starts.
        self.examples = load_examples(config) if config.training.sample_every else []

    def after_epoch(self, epoch, models):
        """epoch counts from 1, as in training.jsonl."""
        training = self.config.training
        last = epoch == training.epochs
        if training.plot_every and (epoch % training.plot_every == 0 or last):
            _reported(self.write_losses, epoch)
        if training.sample_every and (epoch % training.sample_every == 0 or epoch == 1 or last):
            _reported(self.write_samples, epoch, models)

    def title(self, epoch):
        config = self.config
        return (f"{self.run.resolve().name} · {config.framework}/{config.mode} · "
                f"epoch {epoch}/{config.training.epochs}")

    def write_losses(self, epoch):
        records = {}
        for line in (self.run / "training.jsonl").read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["epoch"]] = record  # A repeated epoch keeps its latest record.
        plot_losses([records[key] for key in sorted(records)], self.root / "losses.png",
                    self.title(epoch))

    def write_samples(self, epoch, models):
        config, started = self.config, time.perf_counter()
        steps = config.training.sample_steps or config.steps
        # The learned variance when the model has one; base models generate the image only.
        preview = replace(config,
                          inference=replace(config.inference,
                                            steps=steps,
                                            analyses=[],
                                            uncertainty="none"
                                            if config.mode == "base" else "propagated"))
        rows = []
        for example in self.examples:
            condition, target = prepare_batch(example, config, require_target=True)
            with preserved_random_state(config.device):
                seed_everything(config.seed)  # The same noise at every epoch.
                result = sample(condition, models, preview)
            rows.append(dict(name=example["name"], panels=self.panels(condition, target, result)))
        where = " · central slice of the last axis" if config.model.spatial_dims == 3 else ""
        plot_samples(rows, self.root / "samples.png",
                     f"{self.title(epoch)} · {steps} sampling steps{where}")
        print(f"Monitor: samples.png updated in {time.perf_counter() - started:.0f} s", flush=True)

    def panels(self, condition, target, result):
        """Inputs, target, generated image, absolute error and, if any, predicted std."""
        kwargs = self.config.data.kwargs
        truth, generated = _plane(target[0, 0]), _plane(result.image[0, 0])
        low, high = float(truth.min()), float(truth.max())
        # The generated image shares the target's gray scale, so saturation shows.
        limits = (low, high if high > low else low + 1.0)
        inputs = _labels(kwargs.get("condition"), condition.shape[1], "input")
        panels = [
            dict(title=label, image=_plane(condition[0, index]), limits=None, heat=False)
            for index, label in enumerate(inputs)
        ]
        panels += [
            dict(title=_labels(kwargs.get("target"), target.shape[1], "target")[0],
                 image=truth,
                 limits=limits,
                 heat=False),
            dict(title="generated", image=generated, limits=limits, heat=False)
        ]
        error = np.abs(generated - truth)
        panels.append(
            dict(title="|generated − target|", image=error, limits=(0.0, _upper(error)), heat=True))
        if result.variance is not None:
            deviation = np.sqrt(np.maximum(_plane(result.variance[0, 0]), 0.0))
            panels.append(
                dict(title="predicted std",
                     image=deviation,
                     limits=(0.0, _upper(deviation)),
                     heat=True))
        return panels


def _reported(action, *args):
    """A monitoring failure must not end a training run: report it and carry on."""
    try:
        action(*args)
    except Exception:  # Any failure here is reported, never fatal.
        print(f"WARNING: training monitor ({action.__name__}) failed; training continues.",
              file=sys.stderr,
              flush=True)
        traceback.print_exc()


def _labels(names, count, role):
    """Channel labels from a dataset's modality list, e.g. BraTS's, else by role."""
    if (isinstance(names, (list, tuple)) and len(names) == count
            and all(isinstance(name, str) for name in names)):
        return [f"{role}: {name}" for name in names]
    return [role] if count == 1 else [f"{role} {index}" for index in range(count)]


def _plane(image):
    """2D view of one channel: an image as it is, or the central slice of a volume's
    last axis, turned a quarter so that the slice's second axis points up."""
    array = image.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = np.rot90(array[..., array.shape[-1] // 2])
    return np.ascontiguousarray(array)


def _upper(values):
    """Color-scale maximum that a few extreme pixels cannot wash out."""
    return float(np.percentile(values, 99.5)) or float(values.max()) or 1.0


def _style(axis):
    """Recessive chrome: solid hairline grid, light axes, muted tick labels."""
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side, spine in axis.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
    axis.tick_params(colors=AXIS, labelcolor=MUTED, labelsize=8)


def _save(figure, path):
    """Replace path atomically: a viewer refreshing the file never reads half an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=110, facecolor=SURFACE)
    os.replace(temporary, path)


def plot_losses(records, path, title):
    """One panel per recorded quantity against the epoch; never two scales on one axis.
    Quantities that stay positive use a log scale."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    series = {}
    for record in records:
        for key, value in record.items():
            if key not in SKIPPED and isinstance(value, (int, float)):
                series.setdefault(key, []).append((record["epoch"], value))
    if not series:
        return
    columns = 2 if len(series) == 4 else min(len(series), 3)
    rows = -(-len(series) // columns)
    figure = Figure(figsize=(4.4 * columns, 3.1 * rows + 0.4),
                    layout="constrained",
                    facecolor=SURFACE)
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for axis, (key, points) in zip(axes, series.items()):
        epochs, values = zip(*points)
        axis.plot(epochs,
                  values,
                  color=SERIES,
                  linewidth=1.4,
                  solid_capstyle="round",
                  solid_joinstyle="round",
                  marker="o" if len(values) <= 40 else "",
                  markersize=5.5,
                  markeredgecolor=SURFACE,
                  markeredgewidth=1.3)
        # End dot on the latest value, which the right-hand title states.
        axis.plot(epochs[-1:],
                  values[-1:],
                  linestyle="",
                  marker="o",
                  markersize=7,
                  color=SERIES,
                  markeredgecolor=SURFACE,
                  markeredgewidth=1.3)
        if min(values) > 0:
            axis.set_yscale("log")
        _style(axis)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("epoch", color=SECONDARY, fontsize=9)
        axis.set_title(TITLES.get(key, key), loc="left", color=INK, fontsize=10)
        axis.set_title(f"last {values[-1]:.4g}", loc="right", color=SECONDARY, fontsize=9)
    for axis in axes[len(series):]:
        axis.set_visible(False)
    figure.suptitle(title, color=INK, fontsize=11)
    _save(figure, path)


def plot_samples(rows, path, title):
    """One row per example and one column per view; error and std carry color bars."""
    from matplotlib.figure import Figure

    columns = max(len(row["panels"]) for row in rows)
    figure = Figure(figsize=(2.4 * columns, 2.5 * len(rows) + 0.4),
                    layout="constrained",
                    facecolor=SURFACE)
    grid = figure.subplots(len(rows), columns, squeeze=False)
    for index, (row, axes) in enumerate(zip(rows, grid)):
        for axis, panel in zip(axes, row["panels"]):
            low, high = panel["limits"] or (None, None)
            shown = axis.imshow(panel["image"],
                                cmap=HEAT if panel["heat"] else "gray",
                                vmin=low,
                                vmax=high,
                                interpolation="nearest")
            if panel["heat"]:
                bar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
                bar.outline.set_visible(False)
                bar.ax.tick_params(colors=AXIS, labelcolor=MUTED, labelsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if index == 0:
                axis.set_title(panel["title"], color=INK, fontsize=9)
        axes[0].set_ylabel(row["name"], color=SECONDARY, fontsize=9)
        for axis in axes[len(row["panels"]):]:
            axis.set_visible(False)
    figure.suptitle(title, color=INK, fontsize=10)
    _save(figure, path)
