from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def flatten_config(cfg: Dict[str, Any], sections=("run", "model", "data", "training", "inference")) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section in sections:
        values = cfg.get(section, {})
        if isinstance(values, dict):
            flat.update(values)
    return flat


def merge_into_namespace(cli: argparse.Namespace, cfg: Dict[str, Any], sections=("run", "model", "data", "training", "inference")) -> argparse.Namespace:
    flat = flatten_config(cfg, sections=sections)
    for key, value in flat.items():
        if getattr(cli, key, None) is None:
            setattr(cli, key, value)
    return cli


def default(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def dump_resolved_config(args: argparse.Namespace, output_path: str | Path) -> None:
    if yaml is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)
