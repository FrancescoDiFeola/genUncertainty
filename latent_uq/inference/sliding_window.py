from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from latent_uq.frameworks import is_latent_framework, normalize_framework
from src.inference.utils import (
    map_correlations_multi_thresholds,
    norm_percentile,
    random_sparsification_fast,
    sparsification_curve_fast,
    summarize_uncertainty,
    uncertainty_error_tail_bins_torch,
)



def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _encode(autoencoder, image: torch.Tensor) -> torch.Tensor:
    ae = _unwrap(autoencoder)
    encoded = ae(image)
    if isinstance(encoded, (tuple, list)) and len(encoded) >= 2:
        return encoded[1]
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.sample()
    raise RuntimeError("Unsupported autoencoder encode output for sliding-window inference.")


def _decode(autoencoder, latent: torch.Tensor) -> torch.Tensor:
    ae = _unwrap(autoencoder)
    decoded = ae.decode(latent)
    if isinstance(decoded, (tuple, list)):
        return decoded[0]
    if hasattr(decoded, "sample"):
        return decoded.sample
    return decoded


def _model_forward(model, x: torch.Tensor, timesteps: torch.Tensor, context=None):
    out = model(x=x, timesteps=timesteps, context=context)
    if isinstance(out, (tuple, list)):
        return out[0], out[1] if len(out) > 1 else None
    if isinstance(out, dict):
        pred = out.get("prediction", out.get("sample", out.get("pred")))
        return pred, out.get("logvar")
    return out, None


def _batch_timestep(t, batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(t, device=device).reshape(-1)
    if value.numel() == 0:
        raise ValueError("Empty scheduler timestep.")
    return value[:1].repeat(batch_size).long()


def _sample_patch(
    condition_patch: torch.Tensor,
    *,
    args: Any,
    model,
    autoencoder,
    context_encoder,
    scheduler,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one patch and return pixel-space prediction and uncertainty."""
    framework = normalize_framework(args.framework)
    mode = str(args.mode)
    device = condition_patch.device
    latent = is_latent_framework(framework)

    if latent:
        if autoencoder is None:
            raise RuntimeError("Latent patch inference requires an autoencoder.")
        with torch.no_grad():
            condition_model = _encode(autoencoder, condition_patch)
        condition_model = condition_model * float(scaling_factor)
    else:
        condition_model = condition_patch

    x = torch.randn_like(condition_model)
    uncertainty_sum = torch.zeros_like(x[:, :1])
    valid_uncertainty_steps = 0
    last_logvar: Optional[torch.Tensor] = None
    previous_uncertainty: Optional[torch.Tensor] = None

    if framework in {"dm", "ldm"}:
        scheduler.set_timesteps(int(args.num_inference_steps))
        if hasattr(scheduler, "alphas_cumprod"):
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
        timesteps = scheduler.timesteps
    else:
        # RFlowScheduler timesteps depend on the current model-space patch size.
        scheduler.set_timesteps(
            num_inference_steps=int(args.num_inference_steps),
            device=device,
            input_img_size_numel=int(condition_model.shape[-2] * condition_model.shape[-1]),
        )
        timesteps = scheduler.timesteps

    next_timesteps = None
    if framework in {"fm", "lfm"}:
        next_timesteps = torch.cat(
            [timesteps[1:], torch.zeros(1, dtype=timesteps.dtype, device=timesteps.device)]
        )

    num_steps = len(timesteps)
    k_steps = max(1, min(int(getattr(args, "K", 10)), max(num_steps - 1, 1)))

    for index, timestep in enumerate(timesteps):
        t_batch = _batch_timestep(timestep, x.shape[0], device)
        context = None
        if mode == "selfcond" and context_encoder is not None:
            if previous_uncertainty is None:
                previous_uncertainty = torch.zeros_like(x[:, :1])
            context = context_encoder(previous_uncertainty)

        model_input = torch.cat([x, condition_model], dim=1)
        prediction, logvar = _model_forward(model, model_input, t_batch, context=context)
        last_logvar = logvar

        if logvar is not None and (num_steps - k_steps - 1) <= index < (num_steps - 1):
            variance = torch.exp(logvar.float())
            if framework in {"dm", "ldm"}:
                alpha_bar = scheduler.alphas_cumprod[t_batch[0]].reshape(1, 1, 1, 1)
                variance = ((1.0 - alpha_bar) / (alpha_bar + 1e-8)) * variance
            uncertainty_sum = uncertainty_sum + variance.mean(dim=1, keepdim=True)
            valid_uncertainty_steps += 1
            previous_uncertainty = variance.detach().mean(dim=1, keepdim=True)

        if framework in {"dm", "ldm"}:
            stepped = scheduler.step(prediction, t_batch, x)
        else:
            stepped = scheduler.step(prediction, timestep, x, next_timesteps[index])
        x = stepped[0] if isinstance(stepped, (tuple, list)) else getattr(stepped, "prev_sample", stepped)

    uncertainty_model = uncertainty_sum / max(valid_uncertainty_steps, 1)
    if mode == "base" or last_logvar is None:
        uncertainty_model = torch.zeros_like(x[:, :1])

    if not latent:
        return x, uncertainty_model

    prediction_pixel = _decode(autoencoder, x / float(scaling_factor))
    if mode == "base" or last_logvar is None:
        return prediction_pixel, torch.zeros_like(prediction_pixel[:, :1])

    # Match the legacy latent uncertainty propagation: perturb the final latent
    # according to the accumulated variance and estimate pixel-space variance.
    sigma = torch.sqrt(uncertainty_model.clamp_min(1e-12))
    if sigma.shape[1] != x.shape[1]:
        sigma = sigma.expand(-1, x.shape[1], -1, -1)
    decoded = []
    for _ in range(int(getattr(args, "mc_decode_samples", 10))):
        z_sample = x + sigma * torch.randn_like(x)
        decoded.append(_decode(autoencoder, z_sample / float(scaling_factor)))
    uncertainty_pixel = torch.stack(decoded, dim=0).var(dim=0, unbiased=False).mean(dim=1, keepdim=True)
    return prediction_pixel, uncertainty_pixel


@torch.no_grad()
def sliding_window_generate(
    condition: torch.Tensor,
    *,
    args: Any,
    model,
    autoencoder,
    context_encoder,
    scheduler,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MONAI sliding-window inference with Gaussian stitching.

    Prediction and uncertainty are concatenated so MONAI applies exactly the
    same spatial weighting and stitching to both outputs.
    """
    try:
        from monai.inferers import sliding_window_inference
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("MONAI is required for --patch-based inference.") from exc

    roi = (int(args.patch_size), int(args.patch_size))

    def predictor(window: torch.Tensor) -> torch.Tensor:
        pred, uncertainty = _sample_patch(
            window,
            args=args,
            model=model,
            autoencoder=autoencoder,
            context_encoder=context_encoder,
            scheduler=scheduler,
            scaling_factor=scaling_factor,
        )
        if uncertainty.shape[-2:] != pred.shape[-2:]:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([pred, uncertainty], dim=1)

    stitched = sliding_window_inference(
        inputs=condition,
        roi_size=roi,
        sw_batch_size=int(getattr(args, "sw_batch_size", 1)),
        predictor=predictor,
        overlap=float(args.patch_overlap),
        mode=str(getattr(args, "patch_blend_mode", "gaussian")),
        sigma_scale=float(getattr(args, "patch_sigma_scale", 0.125)),
        padding_mode="constant",
        cval=float(args.patch_pad_value),
        sw_device=condition.device,
        device=condition.device,
        progress=bool(getattr(args, "patch_progress", False)),
    )
    output_channels = int(getattr(args, "out_ch", 1))
    prediction = stitched[:, :output_channels]
    uncertainty = stitched[:, output_channels:output_channels + 1]
    return prediction, uncertainty


def _safe_data_range(gt: np.ndarray) -> float:
    value = float(np.nanmax(gt) - np.nanmin(gt))
    return value if np.isfinite(value) and value > 0 else 1.0


def _write_metrics(csv_writer, sample_id: int, gt: np.ndarray, pred: np.ndarray, unc: Optional[np.ndarray]):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    row: Dict[str, Any] = {
        "Sample": sample_id,
        "MSE": float(np.mean((gt - pred) ** 2)),
        "PSNR": float(peak_signal_noise_ratio(gt, pred, data_range=_safe_data_range(gt))),
        "SSIM": float(structural_similarity(gt, pred, data_range=_safe_data_range(gt))),
    }
    if unc is not None:
        unc_tensor = torch.as_tensor(unc)[None, None]
        unc_norm = norm_percentile(unc_tensor)[0, 0].numpy()
        norm_corr = map_correlations_multi_thresholds(unc_norm, pred, gt)
        raw_corr = map_correlations_multi_thresholds(unc, pred, gt)
        row.update({
            "Pearson_u_norm": norm_corr["pearson"],
            "Spearman_u_norm": norm_corr["spearman"],
            "AUROC_top15_u_norm": norm_corr["AUROC_top15"],
            "AUROC_top10_u_norm": norm_corr["AUROC_top10"],
            "AUROC_top5_u_norm": norm_corr["AUROC_top5"],
            "Pearson_u_unnorm": raw_corr["pearson"],
            "Spearman_u_unnorm": raw_corr["spearman"],
            "AUROC_top15_u_unnorm": raw_corr["AUROC_top15"],
            "AUROC_top10_u_unnorm": raw_corr["AUROC_top10"],
            "AUROC_top5_u_unnorm": raw_corr["AUROC_top5"],
        })
    csv_writer.writerow(row)


def _write_analysis(csv_writer, analysis: str, sample_id: int, gt: np.ndarray, pred: np.ndarray, unc: Optional[np.ndarray]):
    error = np.abs(pred - gt)
    if analysis == "metrics":
        _write_metrics(csv_writer, sample_id, gt, pred, unc)
        return
    if unc is None:
        raise RuntimeError(f"Analysis '{analysis}' requires an uncertainty map.")
    if analysis == "sparsification":
        u = unc.reshape(-1)
        e = error.reshape(-1)
        fractions, curve = sparsification_curve_fast(u, e)
        random_curve = random_sparsification_fast(e, fractions)
        order = np.argsort(-e)
        e_sorted = e[order]
        n = len(e_sorted)
        k_vals = np.minimum(np.round(fractions * n).astype(int), n - 1)
        cumsum = np.cumsum(e_sorted)
        total = cumsum[-1]
        remaining = np.where(k_vals == 0, total, total - cumsum[k_vals - 1])
        oracle_curve = remaining / (n - k_vals)
        oracle_curve = oracle_curve / (oracle_curve[0] + 1e-12)
        for f, c, r, o in zip(fractions, curve, random_curve, oracle_curve):
            csv_writer.writerow({"Sample": sample_id, "Fraction": f, "Error": c, "RandomError": r, "OracleError": o})
        return
    if analysis == "spatial_error_correlation":
        stats = summarize_uncertainty(unc)
        csv_writer.writerow({
            "Sample": sample_id,
            "MAE": float(error.mean()),
            "u_mean": stats["u_mean"],
            "u_p95": stats["u_p95"],
            "u_p99": stats["u_p99"],
            "u_top1_mean": stats["u_top1_mean"],
            "top5_u_mean": stats["u_top5_mean"],
        })
        return
    if analysis == "calibration_bins":
        for row in uncertainty_error_tail_bins_torch(unc, error, sample_id):
            csv_writer.writerow(row)
        return
    raise ValueError(f"Unsupported analysis: {analysis}")


def log_and_analyze_stitched_batch(
    *,
    condition: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    uncertainty: torch.Tensor,
    writer,
    step: int,
    csv_writers: Dict[str, Dict[str, Any]],
    mode: str,
) -> None:
    """Run all analyses and TensorBoard visualization on complete stitched images."""
    import matplotlib.pyplot as plt

    cond_cpu = condition.detach().cpu()
    target_cpu = target.detach().cpu()
    pred_cpu = prediction.detach().cpu()
    unc_cpu = uncertainty.detach().cpu()
    unc_vis = norm_percentile(unc_cpu)
    err_vis = norm_percentile(torch.abs(pred_cpu - target_cpu))

    batch_size = target_cpu.shape[0]
    fig, axes = plt.subplots(batch_size, 5, figsize=(15, 3 * batch_size), squeeze=False)
    titles = ["Condition", "Ground truth", "Stitched prediction", "Stitched uncertainty", "Absolute error"]
    for i in range(batch_size):
        gt_np = target_cpu[i, 0].numpy()
        pred_np = pred_cpu[i, 0].numpy()
        unc_np = None if mode == "base" else unc_cpu[i, 0].numpy()
        sample_id = step * batch_size + i
        for analysis, info in csv_writers.items():
            _write_analysis(info["writer"], analysis, sample_id, gt_np, pred_np, unc_np)

        panels = [cond_cpu[i, 0], target_cpu[i, 0], pred_cpu[i, 0], unc_vis[i, 0], err_vis[i, 0]]
        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel.numpy(), cmap="hot" if j in (3, 4) else "gray")
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")
    fig.tight_layout()
    writer.add_figure("Test/SlidingWindow_StitchedInference", fig, global_step=step)
    writer.add_images("Test/stitched_prediction", pred_cpu, global_step=step, dataformats="NCHW")
    if mode != "base":
        writer.add_images("Test/stitched_uncertainty", unc_vis, global_step=step, dataformats="NCHW")
    plt.close(fig)
