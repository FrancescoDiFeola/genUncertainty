from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from latent_uq.frameworks import normalize_framework

REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_SCRIPTS = {
    ("ldm", "base"): "latent_uq/backends/training_backend_scripts/train/train_ldm_base.py",
    ("ldm", "aleatoric"): "latent_uq/backends/training_backend_scripts/train/train_ldm_aleatoric.py",
    ("ldm", "selfcond"): "latent_uq/backends/training_backend_scripts/train/train_ldm_selfcond.py",
    ("lfm", "base"): "latent_uq/backends/training_backend_scripts/train/train_lfm_base.py",
    ("lfm", "aleatoric"): "latent_uq/backends/training_backend_scripts/train/train_lfm_aleatoric.py",
    ("lfm", "selfcond"): "latent_uq/backends/training_backend_scripts/train/train_lfm_selfcond.py",
    ("dm", "base"): "latent_uq/backends/training_backend_scripts/train/train_dm_base.py",
    ("dm", "aleatoric"): "latent_uq/backends/training_backend_scripts/train/train_dm_aleatoric.py",
    ("dm", "selfcond"): "latent_uq/backends/training_backend_scripts/train/train_dm_selfcond.py",
    ("fm", "base"): "latent_uq/backends/training_backend_scripts/train/train_fm_base.py",
    ("fm", "aleatoric"): "latent_uq/backends/training_backend_scripts/train/train_fm_aleatoric.py",
    ("fm", "selfcond"): "latent_uq/backends/training_backend_scripts/train/train_fm_selfcond.py",
}


def cfg_get(args: Any, cfg: Dict[str, Any], name: str, default: Any = None) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    return cfg.get(name, default)


def add_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend([str(v) for v in value])
        return
    cmd.extend([flag, str(value)])


def build_training_backend_command(args: Any, cfg: Dict[str, Any]) -> List[str]:
    framework = normalize_framework(cfg_get(args, cfg, "framework", "ldm"))
    mode = cfg_get(args, cfg, "mode", "base")
    key = (framework, mode)
    if key not in TRAIN_SCRIPTS:
        raise ValueError(f"Unsupported framework/mode combination: {framework}/{mode}")

    cmd: List[str] = [sys.executable, str(REPO_ROOT / TRAIN_SCRIPTS[key])]

    add_arg(cmd, "--experiment_name", cfg_get(args, cfg, "experiment_name"))
    add_arg(cmd, "--task", cfg_get(args, cfg, "task", "custom"))
    add_arg(cmd, "--output_dir", cfg_get(args, cfg, "output_dir", "outputs/checkpoints"))

    add_arg(cmd, "--diff_ckpt", cfg_get(args, cfg, "diff_ckpt"))
    add_arg(cmd, "--VAE_ckpt", cfg_get(args, cfg, "vae_ckpt"))
    add_arg(cmd, "--context_ckpt", cfg_get(args, cfg, "context_ckpt"))
    add_arg(cmd, "--unc_decoder_ckpt", cfg_get(args, cfg, "unc_decoder_ckpt"))

    add_arg(cmd, "--dataset_csv", cfg_get(args, cfg, "dataset_csv"))
    add_arg(cmd, "--annotation_A", cfg_get(args, cfg, "annotation_A"))
    add_arg(cmd, "--annotation_B", cfg_get(args, cfg, "annotation_B"))
    add_arg(cmd, "--dataroot", cfg_get(args, cfg, "dataroot"))
    add_arg(cmd, "--phase", cfg_get(args, cfg, "phase"))
    add_arg(cmd, "--slice_range", cfg_get(args, cfg, "slice_range"))
    add_arg(cmd, "--mri_modalities", cfg_get(args, cfg, "mri_modalities"))
    add_arg(cmd, "--under_sample_dataset", cfg_get(args, cfg, "under_sample_dataset", False))

    add_arg(cmd, "--num_workers", cfg_get(args, cfg, "num_workers", 8))
    add_arg(cmd, "--n_epochs", cfg_get(args, cfg, "n_epochs", 305))
    add_arg(cmd, "--batch_size", cfg_get(args, cfg, "batch_size", 16))
    add_arg(cmd, "--lr", cfg_get(args, cfg, "lr", 1.5e-5))
    add_arg(cmd, "--epoch_start", cfg_get(args, cfg, "epoch_start", 0))
    add_arg(cmd, "--diff_loss_weight", cfg_get(args, cfg, "diff_loss_weight", 1.0))
    add_arg(cmd, "--uncertainty_loss_weight", cfg_get(args, cfg, "uncertainty_loss_weight"))

    add_arg(cmd, "--in_ch", cfg_get(args, cfg, "in_ch", 2))
    add_arg(cmd, "--out_ch", cfg_get(args, cfg, "out_ch", 1))
    add_arg(cmd, "--spatial_enc_channels", cfg_get(args, cfg, "spatial_enc_channels"))
    add_arg(cmd, "--uncertainty_calibration", cfg_get(args, cfg, "uncertainty_calibration", False))
    return cmd
