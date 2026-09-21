"""Create reproducible synthetic denoising pairs for the one example dataset."""
import argparse
import csv
from pathlib import Path
import numpy as np


def make_example(output_dir, count=4, size=32, seed=0):
    if count < 1 or size < 8:
        raise ValueError("count must be >= 1 and size >= 8")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Example directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    with (output / "pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "target", "case_id"])
        writer.writeheader()
        for index in range(count):
            cx, cy = rng.uniform(-0.4, 0.4, 2)
            target = (2 * np.exp(-((x - cx)**2 + (y - cy)**2) / 0.15) - 1).astype(np.float32)
            condition = np.clip(target + rng.normal(0, 0.1, target.shape), -1, 1).astype(np.float32)
            np.save(output / f"condition_{index}.npy", condition)
            np.save(output / f"target_{index}.npy", target)
            writer.writerow(
                dict(condition=f"condition_{index}.npy",
                     target=f"target_{index}.npy",
                     case_id=f"example_{index}"))
    return output / "pairs.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="examples/data")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    print(make_example(args.output_dir, args.count, args.size))
