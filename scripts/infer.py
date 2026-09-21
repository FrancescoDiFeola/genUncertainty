"""Repository entry point; the installed equivalent is `latent-uq infer`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_uq.cli import main

if __name__ == "__main__":
    main(["infer", *sys.argv[1:]])
