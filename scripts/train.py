#!/usr/bin/env python3
"""Unified task-agnostic training entrypoint.

Default behaviour uses the generic training loop, which builds datasets from
`data.dataset_class` and supports custom backbones, VAEs and losses via YAML.

For exact historical project-specific training scripts, set:

training:
  backend: legacy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latent_uq.config import flatten_config, merge_into_namespace, read_yaml
from latent_uq.frameworks import normalize_framework
from latent_uq.backends.adapters.training_commands import REPO_ROOT as BACKEND_REPO_ROOT
from latent_uq.backends.adapters.training_commands import build_training_backend_command
from latent_uq.training import run_generic_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified training for DM/FM/LDM/LFM models.")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--framework", choices=["ldm", "lfm", "dm", "ddpm", "diffusion", "fm", "rf", "flow", "flow_matching"], default=None)
    p.add_argument("--mode", choices=["base", "aleatoric", "selfcond"], default=None)
    p.add_argument("--training-backend", dest="backend", choices=["generic", "legacy"], default=None)
    p.add_argument("--experiment-name", dest="experiment_name", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--output-dir", dest="output_dir", type=str, default=None)

    # Model/checkpoint options
    p.add_argument("--diff-ckpt", dest="diff_ckpt", type=str, default=None)
    p.add_argument("--vae-ckpt", dest="vae_ckpt", type=str, default=None)
    p.add_argument("--context-ckpt", dest="context_ckpt", type=str, default=None)
    p.add_argument("--unc-decoder-ckpt", dest="unc_decoder_ckpt", type=str, default=None)
    p.add_argument("--in-ch", dest="in_ch", type=int, default=None)
    p.add_argument("--out-ch", dest="out_ch", type=int, default=None)
    p.add_argument("--spatial-enc-channels", dest="spatial_enc_channels", type=int, default=None)
    p.add_argument("--cross-attention-dim", dest="cross_attention_dim", type=int, default=None)

    # Dataset CLI overrides for built-in datasets. Custom datasets are usually
    # configured in YAML through data.dataset_class and data.dataset_kwargs.
    p.add_argument("--dataset-class", dest="dataset_class", type=str, default=None)
    p.add_argument("--dataset-csv", dest="dataset_csv", type=str, default=None)
    p.add_argument("--annotation-A", dest="annotation_A", type=str, default=None)
    p.add_argument("--annotation-B", dest="annotation_B", type=str, default=None)
    p.add_argument("--dataroot", type=str, default=None)
    p.add_argument("--phase", type=str, default=None)
    p.add_argument("--slice-range", dest="slice_range", type=int, nargs=2, default=None)
    p.add_argument("--mri-modalities", dest="mri_modalities", nargs="+", default=None)
    p.add_argument("--under-sample-dataset", dest="under_sample_dataset", action="store_true")

    # Training options
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--n-epochs", dest="n_epochs", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epoch-start", dest="epoch_start", type=float, default=None)
    p.add_argument("--diff-loss-weight", dest="diff_loss_weight", type=float, default=None)
    p.add_argument("--uncertainty-loss-weight", dest="uncertainty_loss_weight", type=float, default=None)
    p.add_argument("--uncertainty-calibration", dest="uncertainty_calibration", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Validate config and print what would be executed")
    return p.parse_args()


def _resolve_args(cli: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    args = merge_into_namespace(cli, cfg, sections=("run", "model", "data", "training"))
    if getattr(args, "framework", None) is not None:
        args.framework = normalize_framework(args.framework)
    else:
        args.framework = "dm"
    if getattr(args, "mode", None) is None:
        args.mode = "base"
    if getattr(args, "backend", None) is None:
        args.backend = "generic"
    if getattr(args, "task", None) is None:
        args.task = "custom"
    if getattr(args, "output_dir", None) is None:
        args.output_dir = "outputs/checkpoints"
    return args


def main() -> None:
    cli = parse_args()
    raw_cfg = read_yaml(cli.config)
    args = _resolve_args(cli, raw_cfg)

    if args.backend == "legacy":
        flat_cfg = flatten_config(raw_cfg, sections=("run", "model", "data", "training"))
        cmd = build_training_backend_command(args, flat_cfg)
        print("\nResolved legacy training command:\n")
        print(" ".join(cmd))
        print()
        if args.dry_run:
            return
        subprocess.run(cmd, cwd=str(BACKEND_REPO_ROOT), check=True)
        return

    run_generic_training(args, raw_cfg)


if __name__ == "__main__":
    main()
