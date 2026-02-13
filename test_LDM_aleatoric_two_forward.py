import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from torchmetrics.functional import auroc
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import csv
from src.brlp import networks
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.VAE.utils.checkpoints_utils import load_checkpoint

# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


def map_correlations_multi_thresholds(unc_map, pred, gt, percentiles=(95, 90, 85)):
    """
    Compute correlation and failure-discrimination metrics between uncertainty and error maps.

    For each percentile p:
      - define failure pixels as top (100 - p)% highest-error pixels
      - compute AUROC of uncertainty predicting failure

    Args:
        unc_map (np.ndarray): uncertainty map (H, W) or (C, H, W)
        pred (np.ndarray): prediction
        gt (np.ndarray): ground truth
        percentiles (tuple): percentiles defining error thresholds
                             (95 -> top 5%, 90 -> top 10%, etc.)

    Returns:
        results (dict): dictionary with:
            - pearson
            - spearman
            - auroc_top_5
            - auroc_top_10
            - auroc_top_15
    """

    # --- error map ---
    err = np.abs(pred - gt)

    # flatten
    u = unc_map.flatten()
    e = err.flatten()

    # remove NaN / Inf
    mask = np.isfinite(u) & np.isfinite(e)
    u = u[mask]
    e = e[mask]

    results = {}

    # --- global correlations ---
    results["pearson"] = pearsonr(u, e)[0]
    results["spearman"] = spearmanr(u, e)[0]

    # --- failure discrimination at multiple thresholds ---
    for p in percentiles:
        err_thresh = np.percentile(e, p)
        err_bin = (e > err_thresh).astype(np.int32)

        # AUROC is only valid if both classes exist
        if len(np.unique(err_bin)) > 1:
            auroc = roc_auc_score(err_bin, u)
        else:
            auroc = np.nan

        results[f"AUROC_top{100-p}"] = auroc

    return results

@torch.no_grad()
def propagate_uncertainty_eps_to_x0(
    pred_logvar_eps: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
):
    """
    Analytic uncertainty propagation from epsilon-space to x0-space.

    Args:
        pred_logvar_eps: (B, C, H, W) log variance predicted for epsilon
        timesteps:      (B,) or scalar tensor of timesteps
        scheduler:      DDPM/DDIM scheduler with alphas_cumprod

    Returns:
        sigma_x0: (B, C, H, W) propagated std of x0
    """
    device = pred_logvar_eps.device

    # scheduler.alphas_cumprod[t] = \bar{alpha}_t
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Make sure t has shape (B,)
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)

    a_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)  # (B,1,1,1)

    # Convert log-variance → variance
    sigma2_eps = torch.exp(pred_logvar_eps)

    # Propagation: Var(x0) = ((1 - a_bar) / a_bar) * Var(eps)
    sigma2_x0 = (1.0 - a_bar) / (a_bar + 1e-8) * sigma2_eps

    # Return std for visualization
    sigma_x0 = torch.sqrt(torch.clamp(sigma2_x0, min=1e-12))

    return sigma_x0


def norm_percentile(x, pmin=1, pmax=99):
    x = x.clone().to(torch.float32)
    B = x.shape[0]
    normed = torch.zeros_like(x)
    for i in range(B):
        x_i = x[i]
        min_val = torch.quantile(x_i, pmin / 100.0)
        max_val = torch.quantile(x_i, pmax / 100.0)
        x_i = torch.clamp(x_i, min=min_val, max=max_val)
        normed[i] = (x_i - min_val) / (max_val - min_val + 1e-8)
    return normed

@torch.no_grad()
def run_inference_and_log(  # in this function the each sampling step the first forward is used to obtain the uncertainty map without conditioning with cross attention
        diffusion_model,
        context_encoder,
        channels,
        dir,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        csv_writer
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    uncertainty = None
    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.tensor([t], device=device).long()

        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        with autocast(enabled=True):
            dummy_context = torch.zeros((B, 1, 128), device=device)
            pred_error_1, pred_logvar_1 = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=dummy_context
            )

            # Compute normalized uncertainty map for context
            uncertainty_map = torch.exp(pred_logvar_1)
            norm_uncertainty_map = norm_percentile(uncertainty_map)

            if channels == 2:
                # Concatenate predicted error and normalized uncertainty for conditioning
                context_input = torch.cat([norm_percentile(pred_error_1), norm_uncertainty_map], dim=1)  # (B, 2, H, W). # norm_uncertainty_map
            else:
                context_input = norm_uncertainty_map

            context_vector = context_encoder(context_input)  # (B, 1, 128)

        # -----------------------------
        # 🔁 Second Pass: conditioned refinement
        # -----------------------------
        with autocast(enabled=True):
            pred_error_2, pred_logvar_2 = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=context_vector
            )

        # Second-pass uncertainty (store + save PNG)
        second_pass_uncertainty = torch.exp(pred_logvar_2).detach()
        norm_second = norm_percentile(second_pass_uncertainty).cpu()  # (B,1,H,W)

        # Save each sample's uncertainty as PNG
        if int(step) == 1:
            for b in range(B):
                arr = norm_second[b].squeeze(0).numpy()
                png_path = os.path.join(dir, f"sample{b}_{step}_t_step{int(t)}_epoch_350.png")
                plt.imsave(png_path, arr, cmap='hot')

        # -----------------------------
        # 🧮 DDIM step update
        # -----------------------------
        x, _ = scheduler.step(pred_error_2, t_tensor, x)
        uncertainty = pred_logvar_2  # store last uncertainty estimate

    pred_denoised = x
    uncertainty_map = torch.exp(uncertainty)


    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    fig, axes = plt.subplots(nrows=B, ncols=5, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        mask = gt_array != 0

        # Compute metrics
        psnr = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array[mask] - pred_array[mask]) ** 2)
        # psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # mse = np.mean((gt_array - pred_array) ** 2)
        print(psnr, ssim, mse)
        csv_writer.writerow({'Sample': step * B + i, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim})

        for j in range(5):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()

@torch.no_grad()
def run_inference_and_log_v2( # in this function the uncertainty is used for iterative refinement with a two-forward strategy
        diffusion_model,
        autoencoder,
        context_encoder,
        channels,
        dir,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        scaling,
        csv_writer
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    uncertainty = None
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM Sampling")):
        t_tensor = torch.tensor([t], device=device).long()

        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        with autocast(enabled=True):
            if i == 0:
                dummy_context = torch.zeros((B, 1, 128), device=device)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=dummy_context
                )
            else:
                if channels == 2:
                    # Concatenate predicted error and normalized uncertainty for conditioning
                    context_input = torch.cat([norm_percentile(pred_error_2), norm_second_pass], dim=1)  # (B, 2, H, W). # norm_uncertainty_map
                else:
                    context_input = norm_second_pass

                context_vector = context_encoder(context_input)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )
            # Compute normalized uncertainty map for context
            uncertainty_map = torch.exp(pred_logvar_1)
            norm_uncertainty_map = norm_percentile(uncertainty_map)

            if channels == 2:
                # Concatenate predicted error and normalized uncertainty for conditioning
                context_input = torch.cat([norm_percentile(pred_error_1), norm_uncertainty_map], dim=1)  # (B, 2, H, W). # norm_uncertainty_map
            else:
                context_input = norm_uncertainty_map

            context_vector = context_encoder(context_input)  # (B, 1, 128)

        # -----------------------------
        # 🔁 Second Pass: conditioned refinement
        # -----------------------------
        with autocast(enabled=True):
            pred_error_2, pred_logvar_2 = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=context_vector
            )

        # Second-pass uncertainty (store + save PNG)
        second_pass_uncertainty = torch.exp(pred_logvar_2).detach()
        norm_second_pass = norm_percentile(second_pass_uncertainty) # .cpu()  # (B,1,H,W)
        norm_second = norm_percentile(second_pass_uncertainty).cpu()

        # Save each sample's uncertainty as PNG
        if int(step) == 1:
            for b in range(B):
                arr = norm_second[b].squeeze(0).numpy()
                png_path = os.path.join(dir, f"sample{b}_{step}_t_step{int(t)}_epoch_350.png")
                plt.imsave(png_path, arr, cmap='hot')

        # -----------------------------
        # 🧮 DDIM step update
        # -----------------------------
        x, _ = scheduler.step(pred_error_2, t_tensor, x)
        uncertainty = pred_logvar_2  # store last uncertainty estimate

    pred_denoised = x
    uncertainty_map = torch.exp(uncertainty)


    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    fig, axes = plt.subplots(nrows=B, ncols=5, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        mask = gt_array != 0

        # Compute metrics
        psnr = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array[mask] - pred_array[mask]) ** 2)
        # psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # mse = np.mean((gt_array - pred_array) ** 2)
        print(psnr, ssim, mse)
        csv_writer.writerow({'Sample': step * B + i, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim})

        for j in range(5):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()

@torch.no_grad()
def run_inference_and_log_v3( # in this function the uncertainty is used for iterative refinement without two-forward
        diffusion_model,
        autoencoder,
        context_encoder,
        channels,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        scaling,
        csv_writer
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    uncertainty = None
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM Sampling")):
        t_tensor = torch.tensor([t], device=device).long()

        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        with (autocast(enabled=True)):
            if i == 0:
                dummy_context = torch.zeros((B, 1, 128), device=device)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=dummy_context
                )
            else:
                pred_error_1 = pred_error_2
                pred_logvar_1 = pred_logvar_2

            # Compute normalized uncertainty map for context
            uncertainty_map = torch.exp(pred_logvar_1)
            norm_uncertainty_map = norm_percentile(uncertainty_map)

            if channels == 2:
                # Concatenate predicted error and normalized uncertainty for conditioning
                context_input = torch.cat([norm_percentile(pred_error_1), norm_uncertainty_map], dim=1)  # (B, 2, H, W). # norm_uncertainty_map
            else:
                context_input = norm_uncertainty_map

            context_vector = context_encoder(context_input)  # (B, 1, 128)

        # -----------------------------
        # 🔁 Second Pass: conditioned refinement
        # -----------------------------
        with autocast(enabled=True):
            pred_error_2, pred_logvar_2 = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=context_vector
            )

        # Second-pass uncertainty (store + save PNG)
        second_pass_uncertainty = torch.exp(pred_logvar_2).detach()
        norm_second = norm_percentile(second_pass_uncertainty).cpu()

        # Save each sample's uncertainty as PNG
        """
        if int(step) == 1:
            for b in range(B):
                arr = norm_second[b].squeeze(0).numpy()
                png_path = os.path.join(dir, f"sample{b}_{step}_t_step{int(t)}_epoch_350.png")
                plt.imsave(png_path, arr, cmap='hot')
        """
        # -----------------------------
        # 🧮 DDIM step update
        # -----------------------------
        x, _ = scheduler.step(pred_error_2, t_tensor, x)
        uncertainty = pred_logvar_2  # store last uncertainty estimate

    pred_denoised_latent = x
    uncertainty_map_latent = torch.exp(uncertainty)

    pred_denoised = autoencoder.decode(pred_denoised_latent/scaling)
    condition_batch = autoencoder.decode(condition_batch/scaling)

    if uncertainty_map_latent.dim() == 3:
        uncertainty_map_latent = uncertainty_map_latent.unsqueeze(0)

    # Reduce channels → scalar uncertainty
    uncertainty_map_latent = uncertainty_map_latent.mean(dim=1, keepdim=True)

    # Upsample to match decoded resolution
    uncertainty_map = F.interpolate(
        uncertainty_map_latent,
        size=pred_denoised.shape[-2:],  # (H, W) of decoded image
        mode="bilinear",
        align_corners=False,
    )

    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    fig, axes = plt.subplots(nrows=B, ncols=5, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        mask = gt_array != 0

        # Compute metrics
        psnr = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array[mask] - pred_array[mask]) ** 2)
        # psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # mse = np.mean((gt_array - pred_array) ** 2)
        correlations = map_correlations_multi_thresholds(unc[i][0].numpy(), pred_array, gt_array)
        print(f'PSNR: {psnr}, SSIM: {ssim}, MSE: {mse}, Pearson: {correlations["pearson"]}, Spearman: {correlations["spearman"]}, AUROC_top15: {correlations["AUROC_top15"]}, AUROC_top10: {correlations["AUROC_top10"]}, AUROC_top5: {correlations["AUROC_top5"]}')

        csv_writer.writerow({'Sample': step * B + i,
                             'MSE': mse,
                             'PSNR': psnr,
                             'SSIM': ssim,
                             'Pearson': correlations["pearson"],
                             'Spearman': correlations["spearman"],
                             'AUROC_top15': correlations["AUROC_top15"],
                             'AUROC_top10': correlations["AUROC_top10"],
                             'AUROC_top5': correlations["AUROC_top5"],})

        """
        for j in range(5):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')
        """

    # plt.tight_layout()
    # writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    # plt.close()

@torch.no_grad()
def run_inference_and_log_uncertainty_propagation(
        diffusion_model,
        autoencoder,
        context_encoder,
        channels,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
        mc_decode_samples: int = 20,
        K: int = 10,   # number of late denoising steps used for uncertainty aggregation
):
    # ------------------------------------------------------------------
    # Evaluation mode: no dropout, no batchnorm updates
    # ------------------------------------------------------------------
    diffusion_model.eval()
    autoencoder.eval()

    # ------------------------------------------------------------------
    # Basic shapes and setup
    # condition_batch and x live in LATENT space
    # ------------------------------------------------------------------
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # ------------------------------------------------------------------
    # Initial noisy latent z_T ~ N(0, I)
    # This defines a single deterministic diffusion trajectory
    # ------------------------------------------------------------------
    x = torch.randn_like(condition_batch).to(device)

    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)
    num_steps = len(scheduler.timesteps)

    # ==================================================
    # Accumulator for decision-time latent uncertainty
    #
    # U_z0 will store the SUM of propagated variances
    # Var(z0 | t) across the last K denoising steps.
    #
    # This represents uncertainty of the reverse estimator,
    # NOT stochasticity of the diffusion process.
    # ==================================================
    U_z0 = torch.zeros_like(x)
    num_valid_steps = 0

    # ==================================================
    # DDIM sampling in latent space
    #
    # We follow a standard deterministic DDIM trajectory.
    # Uncertainty is accumulated but NEVER injected into x.
    # ==================================================
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM Sampling")):

        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)

        # --------------------------------------------------
        # Reverse model outputs:
        #  - pred_noise: mean estimate of ε
        #  - pred_logvar: learned log-variance of ε
        #
        # The variance head models predictive uncertainty
        # of the reverse estimator at this denoising step.
        # --------------------------------------------------

        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        with (autocast(enabled=True)):
            if i == 0:
                dummy_context = torch.zeros((B, 1, 128), device=device)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=dummy_context
                )
            else:
                pred_error_1 = pred_error_2
                pred_logvar_1 = pred_logvar_2

            # Compute normalized uncertainty map for context
            uncertainty_map = torch.exp(pred_logvar_1)
            norm_uncertainty_map = norm_percentile(uncertainty_map)

            if channels == 2:
                # Concatenate predicted error and normalized uncertainty for conditioning
                context_input = torch.cat([norm_percentile(pred_error_1), norm_uncertainty_map], dim=1)  # (B, 2, H, W). # norm_uncertainty_map
            else:
                context_input = norm_uncertainty_map

            context_vector = context_encoder(context_input)  # (B, 1, 128)

        # -----------------------------
        # 🔁 Second Pass: conditioned refinement
        # -----------------------------
        with autocast(enabled=True):
            pred_error_2, pred_logvar_2 = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=context_vector
            )

        # ==================================================
        # Accumulate late-step decision-time uncertainty
        #
        # We only consider the LAST K steps (excluding final),
        # where the signal-to-noise ratio is high and uncertainty
        # correlates with final reconstruction error.
        #
        # Early steps are ignored as they are noise-dominated.
        # ==================================================
        if (num_steps - K - 1) <= i < (num_steps - 1):

            # --------------------------------------------------
            # ᾱ_t controls how ε uncertainty propagates to z0
            # --------------------------------------------------
            a_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)

            # --------------------------------------------------
            # Learned variance in ε-space
            # This is the ONLY source of uncertainty
            # --------------------------------------------------
            var_eps = torch.exp(pred_logvar_2.float())

            # --------------------------------------------------
            # Analytic propagation from ε-space to z0-space:
            #
            # Var(z0 | t) = (1 - ᾱ_t) / ᾱ_t · Var(ε)
            #
            # This follows directly from the DDPM formulation
            # and does NOT involve sampling.
            # --------------------------------------------------
            var_z0_t = (1.0 - a_bar) / (a_bar + 1e-8) * var_eps

            # --------------------------------------------------
            # If variance head is single-channel but latent
            # has multiple channels, broadcast uncertainty.
            # This assumes channel-wise independence.
            # --------------------------------------------------
            if var_z0_t.shape[1] == 1 and x.shape[1] > 1:
                var_z0_t = var_z0_t.expand(-1, x.shape[1], -1, -1)

            # --------------------------------------------------
            # Accumulate uncertainty across late steps
            # --------------------------------------------------
            U_z0 += var_z0_t
            num_valid_steps += 1

        # ==================================================
        # Deterministic DDIM update
        #
        # Note: uncertainty does NOT affect the trajectory.
        # ==================================================
        x, _ = scheduler.step(pred_error_2, t_tensor, x)

    # ======================================================
    # Final latent mean and aggregated uncertainty
    #
    # pred_denoised_latent is the mean prediction μ_z0
    # ======================================================
    pred_denoised_latent = x

    # --------------------------------------------------
    # Average accumulated variance across K steps
    # --------------------------------------------------
    var_z0 = U_z0 / max(num_valid_steps, 1)

    # --------------------------------------------------
    # Convert variance to standard deviation
    #
    # Required because we will SAMPLE latent perturbations.
    # --------------------------------------------------
    sigma_z0 = torch.sqrt(var_z0.clamp_min(1e-12))

    # ==================================================
    # Decode the latent mean to pixel space
    #
    # This is the final reconstructed image.
    # ==================================================
    pred_denoised = autoencoder.decode(pred_denoised_latent / scaling)
    condition_dec = autoencoder.decode(condition_batch / scaling)

    # ==================================================
    # Monte Carlo decoding of learned uncertainty
    #
    # This approximates propagation through the decoder
    # Jacobian without explicitly computing it.
    #
    # z0^(s) = μ_z0 + σ_z0 ⊙ ε_s
    # x^(s)  = D(z0^(s))
    # ==================================================
    decoded_samples = []
    for _ in range(mc_decode_samples):
        eps = torch.randn_like(pred_denoised_latent)
        z0_s = pred_denoised_latent + sigma_z0 * eps
        x_s = autoencoder.decode(z0_s / scaling)
        decoded_samples.append(x_s)

    decoded_samples = torch.stack(decoded_samples, dim=0)

    # --------------------------------------------------
    # Pixel-space uncertainty = variance of decoded samples
    # --------------------------------------------------
    uncertainty_map = decoded_samples.var(dim=0, unbiased=False)

    # ==================================================
    # Metrics & logging
    #
    # Uncertainty is evaluated ONLY here, not during
    # generation or decoding.
    # ==================================================
    ld = condition_dec.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(torch.abs(pred - gt))

    fig, axes = plt.subplots(nrows=B, ncols=5, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()

        psnr = compute_psnr(
            gt_array, pred_array,
            data_range=gt_array.max() - gt_array.min()
        )
        ssim = compute_ssim(
            gt_array, pred_array,
            data_range=gt_array.max() - gt_array.min()
        )
        mse = np.mean((gt_array - pred_array) ** 2)

        correlations_norm = map_correlations_multi_thresholds(
            unc[i][0].numpy(), pred_array, gt_array)
        correlations_unnorm = map_correlations_multi_thresholds(
            uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)

        csv_writer.writerow({
            'Sample': step * B + i,
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim,
            'Pearson_u_norm': correlations_norm["pearson"],
            'Spearman_u_norm': correlations_norm["spearman"],
            'AUROC_top15_u_norm': correlations_norm["AUROC_top15"],
            'AUROC_top10_u_norm': correlations_norm["AUROC_top10"],
            'AUROC_top5_u_norm': correlations_norm["AUROC_top5"],
            'Pearson_u_unnorm': correlations_unnorm["pearson"],
            'Spearman_u_unnorm': correlations_unnorm["spearman"],
            'AUROC_top15_u_unnorm': correlations_unnorm["AUROC_top15"],
            'AUROC_top10_u_unnorm': correlations_unnorm["AUROC_top10"],
            'AUROC_top5_u_unnorm': correlations_unnorm["AUROC_top5"],
        })

        for j in range(5):
            ax = axes[i][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference_LDM_Uncertainty", plt.gcf(), global_step=step)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=False)
    parser.add_argument('--output_dir', type=str,default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", required=False)
    parser.add_argument('--diff_ckpt', type=str, required=False)
    parser.add_argument('--context_ckpt', default=None, type=str)
    parser.add_argument('--VAE_ckpt', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=str)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--spatial_enc_channels', type=int, default=2)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)

    parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999], help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')

    args = parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.task, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Checkpoint directory: {experiment_dir}")

    args.diff_ckpt = os.path.join(experiment_dir, f"diffusion-ep-{args.epoch}.pth")
    args.context_ckpt = os.path.join(experiment_dir, f"spatial_encoder-ep-{args.epoch}.pth")
    args.VAE_ckpt = os.path.join(args.output_dir, args.task, "VAE")

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    scaling_factor = 1
    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',

        )
        scaling_factor = 9.404202

    elif args.task == "CS":
        transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor()
        ])

        dataset = CityscapesColorDataset(
            root=args.dataroot,
            split="train",
            transform=transform,
            target_transform=transform
        )

    elif args.task == "ND":
        transform = transforms.Compose([
            transforms.Resize((272, 480)),
            transforms.ToTensor()
        ])

        dataset = PairedImageDataset(
            csv_path="train.csv",
            root_dir="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/ND_dataset",
            transform_A=transform,
            transform_B=transform
        )

    elif args.task == "CTPET":
        dataset = Mri2DSlicedataset(args)

    elif args.task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
            # annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
            # annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
        )
        scaling_factor = 7.832608

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)

    # -----------------------
    # ✅ Load autoencoder
    # -----------------------
    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(128, 128, 256),
        latent_channels=3,
        num_res_blocks=2,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )
    autoencoder = autoencoder.to(DEVICE)

    # **Load Checkpoints from checkpoint.py**
    _ = load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=args.VAE_ckpt, model_name="autoencoder")
    autoencoder.eval()

    diffusion = networks.init_ddpm_aleatoric_two_forward(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)
    spatial_encoder = networks.init_spatial_context_encoder(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)
        autoencoder = torch.nn.DataParallel(autoencoder)
        spatial_encoder = torch.nn.DataParallel(spatial_encoder)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    writer = SummaryWriter(comment=args.experiment_name)
    csv_path = os.path.join(experiment_dir, f"{args.experiment_name}_metrics_ir_epoch_{args.epoch}.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Sample', 'MSE', 'PSNR', 'SSIM', 'Pearson_u_norm', 'Spearman_u_norm', 'AUROC_top15_u_norm', 'AUROC_top10_u_norm', 'AUROC_top5_u_norm', 'Pearson_u_unnorm', 'Spearman_u_unnorm', 'AUROC_top15_u_unnorm', 'AUROC_top10_u_unnorm', 'AUROC_top5_u_unnorm']
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

        for step, batch in enumerate(loader):
            img_A = batch["A"].to(DEVICE)
            img_B = batch["B"].to(DEVICE)

            with torch.no_grad():
                _, img_A_latent, _ = autoencoder(img_A)

            img_A_latent = img_A_latent * scaling_factor

            run_inference_and_log_uncertainty_propagation(
                diffusion_model=diffusion,
                autoencoder=autoencoder,
                context_encoder=spatial_encoder,
                channels=args.spatial_enc_channels,
                condition_batch=img_A_latent,
                gt_batch=batch['B'],
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                scaling=scaling_factor,
                csv_writer=writer_csv
            )

    print(f"✅ Inference complete. Metrics saved to {csv_path}")
