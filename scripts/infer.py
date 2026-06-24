#!/usr/bin/env python3
"""Unified task-agnostic inference entrypoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latent_uq.config import default, dump_resolved_config, merge_into_namespace, read_yaml
from latent_uq.frameworks import normalize_framework, is_latent_framework
from latent_uq.inference.analysis import (
    PUBLIC_ANALYSES,
    canonicalize_analyses,
    validate_analyses,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified inference for image-level and latent diffusion/flow variants.")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config file.")

    p.add_argument("--framework", choices=["ldm", "lfm", "dm", "ddpm", "diffusion", "fm", "rf", "flow", "flow_matching"], default=None)
    p.add_argument("--mode", choices=["base", "aleatoric", "selfcond"], default=None)
    p.add_argument("--task", type=str, default=None)

    # Preferred interface: enable one or more analyses in the YAML `analysis` section,
    # or pass them from the CLI with `--analyses metrics sparsification`.
    p.add_argument(
        "--analyses",
        nargs="+",
        choices=list(PUBLIC_ANALYSES),
        default=None,
        help="One or more analyses to run. Overrides the YAML analysis section.",
    )
    # Backward-compatible single-analysis option. Old aliases are translated internally.
    p.add_argument(
        "--analysis",
        choices=[
            "metrics",
            "metrics_no_uncertainty",
            "sparsification",
            "uncertainty_eval",
            "spatial_error_correlation",
            "uncertainty_cal",
            "calibration_bins",
        ],
        default=None,
        help="Backward-compatible single-analysis option. Prefer --analyses or YAML analysis flags.",
    )

    p.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    p.add_argument("--checkpoint-root", dest="checkpoint_root", type=str, default=None)
    p.add_argument("--experiment-name", dest="experiment_name", type=str, default=None)
    p.add_argument("--epoch", type=str, default=None)
    p.add_argument("--diff-ckpt", dest="diff_ckpt", type=str, default=None)
    p.add_argument("--context-ckpt", dest="context_ckpt", type=str, default=None)
    p.add_argument("--vae-ckpt", dest="vae_ckpt", type=str, default=None)

    p.add_argument("--dataset-class", dest="dataset_class", type=str, default=None)
    p.add_argument("--annotation-A", dest="annotation_A", type=str, default=None)
    p.add_argument("--annotation-B", dest="annotation_B", type=str, default=None)
    p.add_argument("--csv-path", dest="csv_path", type=str, default=None)
    p.add_argument("--dataroot", type=str, default=None)
    p.add_argument("--output-size", dest="output_size", type=int, default=None)
    p.add_argument("--scaling-factor", dest="scaling_factor", type=float, default=None)
    p.add_argument("--motion-level", dest="motion_level", type=str, default=None)

    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--in-ch", dest="in_ch", type=int, default=None)
    p.add_argument("--out-ch", dest="out_ch", type=int, default=None)
    p.add_argument("--spatial-enc-channels", dest="spatial_enc_channels", type=int, default=None)
    p.add_argument("--cross-attention-dim", dest="cross_attention_dim", type=int, default=None)

    p.add_argument("--num-train-timesteps", dest="num_train_timesteps", type=int, default=None)
    p.add_argument("--num-inference-steps", dest="num_inference_steps", type=int, default=None)
    p.add_argument("--beta-start", dest="beta_start", type=float, default=None)
    p.add_argument("--beta-end", dest="beta_end", type=float, default=None)
    p.add_argument("--beta-schedule", dest="beta_schedule", type=str, default=None)
    p.add_argument("--base-img-size-numel", dest="base_img_size_numel", type=int, default=None)
    p.add_argument("--input-img-size-numel", dest="input_img_size_numel", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _analyses_from_yaml(cfg: Dict[str, Any]) -> List[str]:
    """Read requested analyses from the YAML `analysis` section.

    Supported YAML formats:

    analysis:
      metrics: true
      sparsification: false
      spatial_error_correlation: true
      calibration_bins: false

    or:

    analysis:
      enabled: [metrics, sparsification]
    """
    section = cfg.get("analysis", None)
    if section is None:
        # Backward compatibility with older configs where `run.analysis` was a string.
        run_section = cfg.get("run", {})
        old_value = run_section.get("analysis", None) if isinstance(run_section, dict) else None
        return [old_value] if old_value else ["metrics"]

    if isinstance(section, str):
        return [section]

    if isinstance(section, list):
        return list(section)

    if isinstance(section, dict):
        if "enabled" in section:
            enabled = section["enabled"]
            if isinstance(enabled, str):
                return [enabled]
            return list(enabled)
        requested = []
        for name in PUBLIC_ANALYSES:
            if bool(section.get(name, False)):
                requested.append(name)
        return requested or ["metrics"]

    raise ValueError("Invalid YAML `analysis` section. Use a mapping, list, or string.")


def resolve_analyses(cli: argparse.Namespace, cfg: Dict[str, Any], mode: str) -> List[str]:
    if cli.analyses is not None:
        requested = cli.analyses
    elif cli.analysis is not None:
        requested = [cli.analysis]
    else:
        requested = _analyses_from_yaml(cfg)
    return validate_analyses(mode, canonicalize_analyses(requested))


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.framework = normalize_framework(default(args.framework, "ldm"))
    args.mode = default(args.mode, "aleatoric")
    args.task = default(args.task, "custom")
    args.output_dir = default(args.output_dir, "outputs")
    args.checkpoint_root = default(args.checkpoint_root, "checkpoints")
    args.experiment_name = default(args.experiment_name, f"{args.framework}_{args.mode}")
    args.epoch = default(args.epoch, "latest")
    args.batch_size = int(default(args.batch_size, 1))
    args.num_workers = int(default(args.num_workers, 4))
    args.in_ch = int(default(args.in_ch, 2))
    args.out_ch = int(default(args.out_ch, 1))
    args.spatial_enc_channels = int(default(args.spatial_enc_channels, 2))
    args.cross_attention_dim = int(default(args.cross_attention_dim, 128))
    args.num_train_timesteps = int(default(args.num_train_timesteps, 1000))
    args.num_inference_steps = int(default(args.num_inference_steps, 30 if args.framework in {"lfm", "fm"} else 50))
    args.beta_start = float(default(args.beta_start, 0.0015))
    args.beta_end = float(default(args.beta_end, 0.0205))
    args.beta_schedule = default(args.beta_schedule, "scaled_linear_beta")
    args.base_img_size_numel = int(default(args.base_img_size_numel, 64 * 64))
    args.input_img_size_numel = int(default(args.input_img_size_numel, 64 * 64))
    args.K = int(default(args.K, 30 if args.framework in {"lfm", "fm"} else 10))
    args.seed = int(default(args.seed, 0))
    if args.device is None:
        if getattr(args, "dry_run", False):
            args.device = "cpu"
        else:
            try:
                import torch
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                args.device = "cpu"
    # motion_level is dataset-specific and should remain unset unless explicitly
    # provided by the user/config, e.g. for T1 motion-correction datasets.
    args.motion_level = default(args.motion_level, None)

    exp_dir = Path(args.checkpoint_root) / args.task / args.experiment_name
    if args.diff_ckpt is None and args.epoch != "latest":
        args.diff_ckpt = str(exp_dir / f"diffusion-ep-{args.epoch}.pth")
    if args.context_ckpt is None and args.mode == "selfcond" and args.epoch != "latest":
        args.context_ckpt = str(exp_dir / f"spatial_encoder-ep-{args.epoch}.pth")
    if args.vae_ckpt is None:
        args.vae_ckpt = str(Path(args.checkpoint_root) / args.task / "VAE")
    if args.diff_ckpt is None and getattr(args, "backbone_class", None) is None:
        raise ValueError("Provide diff_ckpt or set epoch/checkpoint_root/experiment_name, or configure model.backbone_class for an initialized custom backbone.")
    return args


def main() -> None:
    cli = parse_args()
    cfg = read_yaml(cli.config)
    args = merge_into_namespace(cli, cfg)
    args = fill_defaults(args)
    args.analyses = resolve_analyses(cli, cfg, args.mode)

    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return

    import torch
    from monai.utils import set_determinism
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from latent_uq.data.factory import build_dataset
    from latent_uq.data.batch import get_condition, get_target
    from latent_uq.backends.adapters.inference_backends import (
        close_csv_writers,
        make_csv_writers,
        run_inference_backend_batch,
    )
    from latent_uq.models.factory import build_autoencoder, build_latent_model
    from latent_uq.schedulers.factory import build_scheduler

    set_determinism(args.seed)
    device = args.device
    print(json.dumps(vars(args), indent=2))

    dataset, scaling_factor = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    autoencoder = build_autoencoder(device, args.vae_ckpt, framework=args.framework, args=args)
    model, context_encoder = build_latent_model(args, device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        if autoencoder is not None:
            autoencoder = torch.nn.DataParallel(autoencoder)
        if context_encoder is not None:
            context_encoder = torch.nn.DataParallel(context_encoder)

    scheduler = build_scheduler(args, device)
    run_name = f"{args.framework}_{args.mode}_{args.task}_ep{args.epoch}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(args, output_dir / "resolved_config.yaml")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"), comment=run_name)
    csv_writers = make_csv_writers(args, output_dir)

    try:
        for step, batch in enumerate(loader):
            img_A = get_condition(batch).to(device)
            img_B = get_target(batch).to(device)
            if is_latent_framework(args.framework):
                if autoencoder is None:
                    raise RuntimeError("Latent frameworks require an autoencoder.")
                with torch.no_grad():
                    _, model_condition, _ = autoencoder(img_A)
                model_condition = model_condition * scaling_factor
            else:
                model_condition = img_A

            for analysis, csv_info in csv_writers.items():
                run_inference_backend_batch(
                    args=args,
                    model=model,
                    autoencoder=autoencoder,
                    context_encoder=context_encoder,
                    img_A_latent=model_condition,
                    img_B=img_B,
                    writer=writer,
                    step=step,
                    device=device,
                    scheduler=scheduler,
                    scaling_factor=scaling_factor,
                    csv_writer=csv_info["writer"],
                    analysis=analysis,
                )
    finally:
        close_csv_writers(csv_writers)
        writer.close()

    paths = {name: str(item["path"]) for name, item in csv_writers.items()}
    print("Inference complete. Results saved to:")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
