from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from latent_uq.frameworks import normalize_framework
from src.inference.utils import initialize_writers
from src.inference.inference_LDM import (
    run_inference_LDM_vanilla_and_log,
    run_inference_LDM_aleatoric_and_log_uncertainty_propagation,
    run_inference_LDM_aleatoric_and_log_uncertainty_propagation_sparsification,
    run_inference_LDM_self_refining_and_log_uncertainty_propagation,
    run_inference_LDM_self_refining_and_log_uncertainty_propagation_sparsification,
    run_inference_LDM_self_refining_and_log_uncertainty_propagation_ablation,
    run_inference_LDM_self_refining_and_log_uncertainty_eval,
    run_inference_LDM_self_refining_and_log_uncertainty_calibration_tail_bins,
)
from src.inference.inference_LFM import (
    run_inference_LFM_vanilla_and_log,
    run_inference_LFM_aleatoric_and_log_uncertainty_propagation,
    run_inference_LFM_aleatoric_and_log_uncertainty_propagation_sparsification,
    run_inference_LFM_self_refining_and_log_uncertainty_propagation,
    run_inference_LFM_self_refining_and_log_uncertainty_propagation_sparsification,
    run_inference_LFM_self_refining_and_log_uncertainty_propagation_ablation,
    run_inference_LFM_self_refining_and_log_uncertainty_eval,
    run_inference_LFM_self_refining_and_log_uncertainty_calibration_tail_bins,
)
from src.inference.inference_ddpm import (
    run_ddpm_vanilla_inference_and_log,
    run_ddpm_aleatoric_inference_and_log_v2,
    run_ddpm_aleatoric_inference_and_log_v2_sparsification,
    run_inference_and_log_v3_clean_unc_integral,
    run_inference_and_log_v3_clean_unc_integral_sparsification,
    run_inference_and_log_v3_clean_unc_integral_ablation,
    run_inference_and_log_v3_clean_uncertainty_eval,
    run_inference_and_log_v3_clean_uncertainty_calibration_tail_bins,
)
from src.inference.inference_RF import (
    run_inference_RF_vanilla,
    run_inference_RF_aleatoric,
    run_inference_RF_aleatoric_sparsification,
    run_inference_RF_self_refining_and_log_v3_clean_unc_integral,
    run_inference_RF_self_refining_and_log_v3_clean_unc_integral_sparsification,
    run_inference_RF_self_refining_and_log_v3_clean_unc_integral_ablation,
    run_inference_and_log_v3_clean_uncertainty_eval as run_inference_RF_uncertainty_eval,
    run_inference_and_log_v3_clean_uncertainty_calibration_tail_bins as run_inference_RF_calibration_tail_bins,
)

from latent_uq.inference.analysis import (
    LEGACY_WRITER_TYPES,
    canonicalize_analysis_name,
    validate_analyses,
)


def _is_motion_dataset(args: Any) -> bool:
    """Return True only for T1 motion-correction datasets/tasks.

    `motion_level` is meaningful only for synthetic T1 motion correction.
    It must not leak into filenames for generic/custom datasets.
    """
    task = str(getattr(args, "task", "") or "").lower()
    dataset_class = str(getattr(args, "dataset_class", "") or "").lower()
    return any(
        token in {task, dataset_class}
        for token in ("t1motion", "t1_motion", "motion_t1")
    )


def analysis_output_name(args: Any, analysis: str) -> str:
    prefix = {
        "metrics": "metrics",
        "sparsification": "sparsification",
        "spatial_error_correlation": "spatial_error_correlation",
        "calibration_bins": "calibration_bins",
    }[analysis]

    name = f"{prefix}_{args.framework}_{args.mode}_{args.task}_ep{args.epoch}"

    motion_level = getattr(args, "motion_level", None)
    if motion_level not in (None, "", "None") and _is_motion_dataset(args):
        name += f"_motion{motion_level}"

    return name + ".csv"


def make_csv_writer(args: Any, output_dir: Path, analysis: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = canonicalize_analysis_name(analysis)
    writer_type = LEGACY_WRITER_TYPES[analysis]
    if args.mode == "base" and analysis == "metrics":
        writer_type = "metrics_no_uncertainty"
    name = analysis_output_name(args, analysis)
    csv_file, writer = initialize_writers(str(output_dir / name), writer_type=writer_type)
    return csv_file, writer, output_dir / name


def make_csv_writers(args: Any, output_dir: Path):
    analyses = validate_analyses(args.mode, args.analyses)
    writers = {}
    for analysis in analyses:
        csv_file, writer, path = make_csv_writer(args, output_dir, analysis)
        writers[analysis] = {"file": csv_file, "writer": writer, "path": path}
    return writers


def close_csv_writers(writers: Dict[str, Dict[str, Any]]) -> None:
    for item in writers.values():
        item["file"].close()


def _run_latent_backend_batch(
    args: Any,
    model,
    autoencoder,
    context_encoder,
    img_A_latent,
    img_B,
    writer,
    step: int,
    device: str,
    scheduler,
    scaling_factor: float,
    csv_writer,
    analysis: str,
):
    common = dict(
        diffusion_model=model,
        autoencoder=autoencoder,
        condition_batch=img_A_latent,
        gt_batch=img_B,
        step=step,
        device=device,
        scheduler=scheduler,
        scaling=scaling_factor,
        csv_writer=csv_writer,
    )
    fw = normalize_framework(args.framework)
    mode = args.mode

    if fw == "ldm" and mode == "base":
        return run_inference_LDM_vanilla_and_log(writer=writer, **common)
    if fw == "lfm" and mode == "base":
        return run_inference_LFM_vanilla_and_log(writer=writer, **common)

    if fw == "ldm" and mode == "aleatoric":
        if analysis == "sparsification":
            return run_inference_LDM_aleatoric_and_log_uncertainty_propagation_sparsification(K=args.K, **common)
        return run_inference_LDM_aleatoric_and_log_uncertainty_propagation(writer=writer, K=args.K, **common)

    if fw == "lfm" and mode == "aleatoric":
        if analysis == "sparsification":
            return run_inference_LFM_aleatoric_and_log_uncertainty_propagation_sparsification(K=args.K, **common)
        return run_inference_LFM_aleatoric_and_log_uncertainty_propagation(writer=writer, K=args.K, **common)

    if mode == "selfcond":
        if context_encoder is None:
            raise ValueError("context_encoder is required for mode='selfcond'")
        if fw == "ldm":
            if analysis == "sparsification":
                return run_inference_LDM_self_refining_and_log_uncertainty_propagation_sparsification(
                    context_encoder=context_encoder, channels=args.spatial_enc_channels, K=args.K, **common
                )
            if analysis == "spatial_error_correlation":
                return run_inference_LDM_self_refining_and_log_uncertainty_eval(
                    context_encoder=context_encoder, writer=writer, channels=args.spatial_enc_channels, **common
                )
            if analysis == "calibration_bins":
                return run_inference_LDM_self_refining_and_log_uncertainty_calibration_tail_bins(
                    context_encoder=context_encoder, writer=writer, channels=args.spatial_enc_channels, **common
                )
            if getattr(args, "ablation", False):
                return run_inference_LDM_self_refining_and_log_uncertainty_propagation_ablation(
                    context_encoder=context_encoder, writer=writer, channels=args.spatial_enc_channels, K=args.K, **common
                )
            return run_inference_LDM_self_refining_and_log_uncertainty_propagation(
                context_encoder=context_encoder, writer=writer, channels=args.spatial_enc_channels, K=args.K, **common
            )
        if fw == "lfm":
            if analysis == "sparsification":
                return run_inference_LFM_self_refining_and_log_uncertainty_propagation_sparsification(
                    context_encoder=context_encoder, K=args.K, **common
                )
            if analysis == "spatial_error_correlation":
                return run_inference_LFM_self_refining_and_log_uncertainty_eval(
                    context_encoder=context_encoder, writer=writer, K=args.K, **common
                )
            if analysis == "calibration_bins":
                return run_inference_LFM_self_refining_and_log_uncertainty_calibration_tail_bins(
                    context_encoder=context_encoder, writer=writer, K=args.K, **common
                )
            if getattr(args, "ablation", False):
                return run_inference_LFM_self_refining_and_log_uncertainty_propagation_ablation(
                    context_encoder=context_encoder, writer=writer, K=args.K, **common
                )
            return run_inference_LFM_self_refining_and_log_uncertainty_propagation(
                context_encoder=context_encoder, writer=writer, K=args.K, **common
            )
    raise RuntimeError(f"Unsupported latent combination: framework={fw}, mode={mode}, analysis={analysis}")


def _run_image_backend_batch(
    args: Any,
    model,
    context_encoder,
    condition_batch,
    img_B,
    writer,
    step: int,
    device: str,
    scheduler,
    csv_writer,
    analysis: str,
):
    fw = normalize_framework(args.framework)
    mode = args.mode
    condition_batch = condition_batch

    if fw == "dm" and mode == "base":
        return run_ddpm_vanilla_inference_and_log(
            diffusion_model=model,
            condition_batch=condition_batch,
            gt_batch=img_B,
            writer=writer,
            step=step,
            device=device,
            scheduler=scheduler,
            csv_writer=csv_writer,
        )
    if fw == "fm" and mode == "base":
        return run_inference_RF_vanilla(
            diffusion_model=model,
            condition_batch=condition_batch,
            gt_batch=img_B,
            writer=writer,
            step=step,
            device=device,
            scheduler=scheduler,
            csv_writer=csv_writer,
        )

    if fw == "dm" and mode == "aleatoric":
        if analysis == "sparsification":
            return run_ddpm_aleatoric_inference_and_log_v2_sparsification(
                diffusion_model=model,
                condition_batch=condition_batch,
                gt_batch=img_B,
                step=step,
                device=device,
                scheduler=scheduler,
                csv_writer=csv_writer,
            )
        return run_ddpm_aleatoric_inference_and_log_v2(
            diffusion_model=model,
            condition_batch=condition_batch,
            gt_batch=img_B,
            writer=writer,
            step=step,
            device=device,
            scheduler=scheduler,
            csv_writer=csv_writer,
            csv_writer_2=csv_writer,
        )

    if fw == "fm" and mode == "aleatoric":
        if analysis == "sparsification":
            return run_inference_RF_aleatoric_sparsification(
                diffusion_model=model,
                condition_batch=condition_batch,
                gt_batch=img_B,
                step=step,
                device=device,
                scheduler=scheduler,
                csv_writer=csv_writer,
                K=args.K,
            )
        return run_inference_RF_aleatoric(
            diffusion_model=model,
            condition_batch=condition_batch,
            gt_batch=img_B,
            writer=writer,
            step=step,
            device=device,
            scheduler=scheduler,
            csv_writer=csv_writer,
            csv_writer_2=csv_writer,
            K=args.K,
        )

    if mode == "selfcond":
        if context_encoder is None:
            raise ValueError("context_encoder is required for mode='selfcond'")
        if fw == "dm":
            common = dict(
                diffusion_model=model,
                context_encoder=context_encoder,
                channels=args.spatial_enc_channels,
                dir=str(getattr(args, "output_dir", "outputs")),
                condition_batch=condition_batch,
                gt_batch=img_B,
                writer=writer,
                step=step,
                device=device,
                scheduler=scheduler,
                csv_writer=csv_writer,
            )
            if analysis == "sparsification":
                return run_inference_and_log_v3_clean_unc_integral_sparsification(
                    diffusion_model=model,
                    context_encoder=context_encoder,
                    dir=str(getattr(args, "output_dir", "outputs")),
                    condition_batch=condition_batch,
                    k_steps=args.K,
                    gt_batch=img_B,
                    step=step,
                    device=device,
                    scheduler=scheduler,
                    csv_writer=csv_writer,
                )
            if analysis == "spatial_error_correlation":
                return run_inference_and_log_v3_clean_uncertainty_eval(**common)
            if analysis == "calibration_bins":
                return run_inference_and_log_v3_clean_uncertainty_calibration_tail_bins(**common)
            if getattr(args, "ablation", False):
                return run_inference_and_log_v3_clean_unc_integral_ablation(**common)
            return run_inference_and_log_v3_clean_unc_integral(csv_writer_2=csv_writer, **common)

        if fw == "fm":
            common = dict(
                diffusion_model=model,
                context_encoder=context_encoder,
                dir=str(getattr(args, "output_dir", "outputs")),
                condition_batch=condition_batch,
                gt_batch=img_B,
                writer=writer,
                step=step,
                device=device,
                scheduler=scheduler,
                csv_writer=csv_writer,
            )
            if analysis == "sparsification":
                return run_inference_RF_self_refining_and_log_v3_clean_unc_integral_sparsification(
                    diffusion_model=model,
                    context_encoder=context_encoder,
                    condition_batch=condition_batch,
                    gt_batch=img_B,
                    step=step,
                    device=device,
                    scheduler=scheduler,
                    csv_writer=csv_writer,
                    K=args.K,
                )
            if analysis == "spatial_error_correlation":
                return run_inference_RF_uncertainty_eval(K=args.K, **common)
            if analysis == "calibration_bins":
                return run_inference_RF_calibration_tail_bins(K=args.K, **common)
            if getattr(args, "ablation", False):
                return run_inference_RF_self_refining_and_log_v3_clean_unc_integral_ablation(
                    csv_writer_2=csv_writer, K=args.K, **common
                )
            return run_inference_RF_self_refining_and_log_v3_clean_unc_integral(
                csv_writer_2=csv_writer, K=args.K, **common
            )

    raise RuntimeError(f"Unsupported image-level combination: framework={fw}, mode={mode}, analysis={analysis}")


def run_inference_backend_batch(
    args: Any,
    model,
    autoencoder,
    context_encoder,
    img_A_latent,
    img_B,
    writer,
    step: int,
    device: str,
    scheduler,
    scaling_factor: float,
    csv_writer,
    analysis: str,
):
    analysis = canonicalize_analysis_name(analysis)
    fw = normalize_framework(args.framework)
    if fw in {"ldm", "lfm"}:
        return _run_latent_backend_batch(
            args=args,
            model=model,
            autoencoder=autoencoder,
            context_encoder=context_encoder,
            img_A_latent=img_A_latent,
            img_B=img_B,
            writer=writer,
            step=step,
            device=device,
            scheduler=scheduler,
            scaling_factor=scaling_factor,
            csv_writer=csv_writer,
            analysis=analysis,
        )
    if fw in {"dm", "fm"}:
        return _run_image_backend_batch(
            args=args,
            model=model,
            context_encoder=context_encoder,
            condition_batch=img_A_latent,
            img_B=img_B,
            writer=writer,
            step=step,
            device=device,
            scheduler=scheduler,
            csv_writer=csv_writer,
            analysis=analysis,
        )
    raise RuntimeError(f"Unsupported framework: {fw}")
