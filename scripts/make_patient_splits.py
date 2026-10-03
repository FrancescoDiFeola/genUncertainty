"""Patient-level folds: no patient keeps scans in more than one split.

BraTS-GLI-XXXXX-YYY names patient XXXXX at timepoint YYY. The original fold files of
folds_brats2021_gli split some patients across train and test (56 in fold 0), which makes
test metrics optimistic. This script rewrites a fold file so that every patient lives in
one split only.

Policies for a patient found in train and in val or test:
    move  move its val/test scans to train (val/test stay clean and get smaller)
    drop  drop its val/test scans (train unchanged)

Usage:
    python scripts/make_patient_splits.py --split-file folds_brats2021_gli/fold0.json \
        --output folds_patient/fold0.json --policy move
"""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_uq.brats import subject_id  # noqa: E402

SPLITS = ("train", "val", "test")


def patient_id(case_id: str) -> str:
    """BraTS-GLI-00009-001 -> BraTS-GLI-00009."""
    raise NotImplementedError


def shared_patients(values: dict) -> dict[str, set[str]]:
    """Patients whose scans appear in more than one split, mapped to those splits."""
    raise NotImplementedError


def rewrite(values: dict, policy: str) -> tuple[dict, dict]:
    """New fold dictionary under `policy`, and a report of moved or dropped case ids.

    Every other key of the fold file is kept; n_train, n_val and n_test are updated when
    present. The report lists, per affected patient, its case ids and where they went.
    """
    raise NotImplementedError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=("move", "drop"), default="move")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} exists; choose another path")
    values = json.loads(Path(args.split_file).read_text())
    shared = shared_patients(values)
    print(f"{len(shared)} patients have scans in more than one split")
    new_values, report = rewrite(values, args.policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(new_values, indent=2))
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print({split: len(new_values.get(split, [])) for split in SPLITS})


if __name__ == "__main__":
    main()
