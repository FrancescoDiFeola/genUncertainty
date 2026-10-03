"""Step 1: the autoencoder ceiling on brain tumor MRI (inference only).

For every subject of the configured split and every representation in the config:
normalize as MAISI does, reorient and pad, round-trip through the representation, restore,
and compare with the original (global metrics, lesions, frequency). For the first
`synthetic.hosts` subjects, analysis C also inserts synthetic lesions before the round
trip. If a segmenter is configured, lesion detection compares its predictions on the
original and on the candidate T1ce. Writes one CSV per table and a run.json.

Usage:
    python scripts/audit_vae.py --config configs/audit_maisi.yaml --output-dir runs/audit/maisi
"""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_uq.audit import compare, io, representations, synthetic  # noqa: E402
from latent_uq.audit.segmenter import Segmenter  # noqa: E402

TABLES = ("global", "lesions", "frequency", "synthetic")


def write_tables(output: Path, tables: dict[str, list[dict]]) -> None:
    """One CSV per non-empty table; the header is the union of the row keys."""
    for name, rows in tables.items():
        if not rows:
            continue
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def plan_synthetic(settings: dict, rng: np.random.Generator) -> list:
    """`per_volume` lesions drawn from the grid of diameters, contrasts and shapes."""
    wall = float(settings.get("ring_thickness_mm", 2.0))
    # A ring needs a cavity: diameters up to twice the wall are spheres only.
    grid = [synthetic.SyntheticLesion(center=(0, 0, 0), diameter_mm=float(diameter),
                                      contrast=float(contrast), shape=shape,
                                      thickness_mm=wall if shape == "ring" else None)
            for diameter in settings["diameters_mm"] for contrast in settings["contrasts"]
            for shape in settings["shapes"] if shape == "sphere" or diameter > 2 * wall]
    return [grid[i] for i in rng.choice(len(grid), settings["per_volume"], replace=False)]


def audit_subject(case_id: str, config: dict, models: list, analyses: tuple[str, ...],
                  rng: np.random.Generator, segmenter: Segmenter | None) -> dict[str, list[dict]]:
    """All rows for one subject and every representation."""
    data = config["data"]
    volume = io.load_volume(data["cache_dir"], case_id, data.get("modality", "t1ce"))
    normalized = io.maisi_normalize(volume.image)
    tables = {name: [] for name in TABLES}
    raw, segmentation_original = None, None
    if segmenter is not None and "lesions" in analyses:
        raw = io.load_modalities(data["cache_dir"], case_id)
        segmentation_original = segmenter(raw["t1"], raw["t1ce"], raw["t2"], raw["t2f"])
        value_range = io.maisi_range(volume.image)
    placed, masks, with_lesions = None, None, None
    if "synthetic" in analyses:
        planned = plan_synthetic(config["synthetic"], rng)
        placed = synthetic.place(volume.brain, volume.labels > 0, planned, rng)
        with_lesions, masks = synthetic.insert(normalized, placed)
    for model in models:
        prepared, layout = io.prepare(normalized, volume.affine)
        reconstruction = io.restore(model(prepared), layout)
        segmentation_candidate = None
        if raw is not None:
            segmentation_candidate = segmenter(raw["t1"],
                                               io.maisi_denormalize(reconstruction, value_range),
                                               raw["t2"], raw["t2f"])
        result = compare.compare(normalized, reconstruction, volume.labels, volume.brain,
                                 case_id=case_id, representation=model.name,
                                 analyses=tuple(a for a in analyses if a != "synthetic"),
                                 segmentation_original=segmentation_original,
                                 segmentation_candidate=segmentation_candidate,
                                 lesion_options=config.get("lesions"))
        for name, rows in result.items():
            tables[name].extend(rows)
        if with_lesions is not None:
            prepared, layout = io.prepare(with_lesions, volume.affine)
            reconstruction = io.restore(model(prepared), layout)
            rows = synthetic.measure(with_lesions, reconstruction, masks, placed)
            tables["synthetic"].extend(
                dict(row, case_id=case_id, representation=model.name) for row in rows)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-subjects", type=int, help="Overrides data.max_subjects")
    parser.add_argument("--analyses", help="Comma-separated subset of " + ",".join(TABLES))
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new output directory")
    analyses = tuple(args.analyses.split(",")) if args.analyses else tuple(config["analyses"])
    unknown = set(analyses) - set(TABLES)
    if unknown:
        raise ValueError(f"Unknown analyses {sorted(unknown)}; choose from {TABLES}")
    data = config["data"]
    cases = io.split_subjects(data["split_file"], data.get("split", "test"),
                              args.max_subjects or data.get("max_subjects"))
    models = [representations.build(spec) for spec in config["representations"]]
    segmenter = Segmenter(**config["segmenter"]) if config.get("segmenter") else None
    hosts = config.get("synthetic", {}).get("hosts", 0)
    rng = np.random.default_rng(config.get("synthetic", {}).get("seed", 0))
    output.mkdir(parents=True)
    tables = {name: [] for name in TABLES}
    started = time.perf_counter()
    for index, case_id in enumerate(cases, 1):
        # Synthetic lesions only in the first `hosts` subjects.
        subject_analyses = analyses if index <= hosts else tuple(
            a for a in analyses if a != "synthetic")
        result = audit_subject(case_id, config, models, subject_analyses, rng, segmenter)
        for name, rows in result.items():
            tables[name].extend(rows)
        print(f"{index}/{len(cases)} {case_id}, {time.perf_counter() - started:.0f} s", flush=True)
        write_tables(output, tables)  # Rewritten after every subject: partial results survive.
    (output / "run.json").write_text(json.dumps(dict(
        config=config, analyses=analyses, subjects=cases,
        representations=[model.name for model in models],
        seconds=time.perf_counter() - started, python=platform.python_version()), indent=2))
    print(f"Done: {len(cases)} subjects in {output}")


if __name__ == "__main__":
    main()
