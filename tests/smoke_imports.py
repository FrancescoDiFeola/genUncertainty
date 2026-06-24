import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latent_uq.frameworks import normalize_framework
from latent_uq.inference.analysis import validate_analyses
from latent_uq.data.factory import import_object
from latent_uq.data.legacy import BUILTIN_DATASETS


def main():
    assert normalize_framework("ddpm") == "dm"
    assert normalize_framework("flow_matching") == "fm"
    assert validate_analyses("base", ["metrics"]) == ["metrics"]

    # Built-in dataset aliases should resolve to importable adapter classes.
    required_aliases = [
        "denoising",
        "ldct",
        "t1t2",
        "t1motion",
        "ctpet",
        "mri2d",
        "cityscapes",
        "nd",
        "mrtoct",
        "cbcttoct",
    ]
    for alias in required_aliases:
        cls = import_object(alias)
        assert isinstance(cls.__name__, str)

    # Ensure every registered alias is importable.
    for alias in BUILTIN_DATASETS:
        cls = import_object(alias)
        assert isinstance(cls.__name__, str)

    print("Smoke imports OK")


if __name__ == "__main__":
    main()
