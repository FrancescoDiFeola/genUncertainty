"""Build the memory-mapped volume cache read by latent_uq.brats.BraTSVolumeDataset (needs nibabel)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_uq.brats import prepare_brats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", required=True, help="Fold JSON listing train/val/test")
    parser.add_argument("--output-dir", required=True, help="Cache directory, shared by all folds")
    parser.add_argument("--data-dir", help="Folder of the NIfTI release; overrides the split file")
    parser.add_argument("--workers", type=int, default=1, help="Parallel subject conversions")
    args = parser.parse_args()
    print(prepare_brats(args.split_file, args.output_dir, data_dir=args.data_dir,
                        workers=args.workers))
