from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
import torch
import os
import torch.nn.functional as F
from src.inference.utils import sparsification_curve_fast, random_sparsification_fast, sparsification_curve, random_sparsification, norm_percentile, collect_calibration_data, map_correlations_multi_thresholds, summarize_uncertainty


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

####################### LDM self refining ###########################

@torch.no_grad()
def run_inference_LDM_self_refining_and_log(  # in this function the each sampling step the first forward is used to obtain the uncertainty map without conditioning with cross attention
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
def run_inference_LDM_self_refining_and_log_v2( # in this function the uncertainty is used for iterative refinement with a two-forward strategy
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
def run_inference_LDM_self_refining_and_log_v3( # in this function the uncertainty is used for iterative refinement without two-forward
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
def run_inference_LDM_self_refining_and_log_uncertainty_propagation(
        diffusion_model,
        autoencoder,
        context_encoder,
        writer,
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

@torch.no_grad()
def run_inference_LDM_self_refining_and_log_uncertainty_eval(
        diffusion_model,
        autoencoder,
        context_encoder,
        writer,
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
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()


    for i in range(B):
        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        # mask = gt_array != 0

        # --- Calibration data ---
        unc_raw = uncertainty_map[i][0].cpu()


        ######### Compute metrics #########

        mae = np.mean(np.abs(gt_array - pred_array))
        uncertainty_summary = summarize_uncertainty(unc_raw)


        csv_writer.writerow({'Sample': step * B + i,
                             'MAE': mae,
                             'u_mean': uncertainty_summary["u_mean"],
                             'u_p95': uncertainty_summary["u_p95"],
                             'u_p99': uncertainty_summary["u_p99"],
                             'u_top1_mean': uncertainty_summary["u_top1_mean"],
                             'top5_u_mean': uncertainty_summary["u_top5_mean"],
                             })

@torch.no_grad()
def run_inference_LDM_self_refining_and_log_uncertainty_propagation_sparsification(
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
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()

    unc_raw = uncertainty_map.cpu().detach()
    err_raw = abs(pred - gt)

    u = unc_raw.flatten()
    e = err_raw.flatten()

    fractions, curve = sparsification_curve(u, e, max_frac=0.95)
    rand_curve = random_sparsification(e, fractions)
    _, curve_oracle = sparsification_curve(e, e, max_frac=0.95)

    for f, c, r, o in zip(fractions, curve, rand_curve, curve_oracle):
        csv_writer.writerow({
            'Sample': step * B + i,
            'Fraction': f,
            'Error': c,
            'RandomError': r,
            'OracleError': o,
        })

@torch.no_grad()
def run_inference_LDM_self_refining_and_log_uncertainty_propagation_ablation(
        diffusion_model,
        autoencoder,
        context_encoder,
        writer,
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

            context_vector = context_encoder(torch.zeros_like(context_input))
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
############################### LDM aleatoric #####################################

@torch.no_grad()
def run_inference_LDM_aleatoric_and_log(
        diffusion_model,
        autoencoder,
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
    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        pred_noise, pred_logvar = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(pred_noise, t_tensor, x)
        uncertainty = pred_logvar

    pred_denoised_latent = x
    # uncertainty_map_latent = torch.exp(uncertainty)

    pred_denoised = autoencoder.decode(pred_denoised_latent/scaling)
    condition_batch = autoencoder.decode(condition_batch/scaling)

    last_t = t_tensor
    uncertainty_map_latent = propagate_uncertainty_eps_to_x0(
        pred_logvar_eps=uncertainty,
        timesteps=last_t,
        scheduler=scheduler,
    )

    # Upsample to match decoded resolution
    uncertainty_map = F.interpolate(
        uncertainty_map_latent,
        size=pred_denoised.shape[-2:],  # (H, W) of decoded image
        mode="bilinear",
        align_corners=False,
    )

    # print(uncertainty_map.shape)

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
        psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array - pred_array) ** 2)
        correlations_norm = map_correlations_multi_thresholds(unc[i][0].cpu().detach().numpy(), pred_array, gt_array)
        correlations_unnorm = map_correlations_multi_thresholds(uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)

        csv_writer.writerow({'Sample': step * B + i,
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

        for j in range(4):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()

@torch.no_grad()
def run_inference_LDM_aleatoric_and_log_uncertainty_propagation(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
        mc_decode_samples: int = 10,
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
    print(f"num_steps: {num_steps}")
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
        pred_noise, pred_logvar = diffusion_model(
            x=model_input,
            timesteps=t_tensor,
            context=None
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
            var_eps = torch.exp(pred_logvar.float())

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
        x, _ = scheduler.step(pred_noise, t_tensor, x)

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
    # Percentile normalization (visualization + metrics)
    # ==================================================
    def norm_percentile(x, pmin=1, pmax=99):
        x = x.clone().to(torch.float32)
        normed = torch.zeros_like(x)
        for i in range(x.shape[0]):
            lo = torch.quantile(x[i], pmin / 100.0)
            hi = torch.quantile(x[i], pmax / 100.0)
            x_i = torch.clamp(x[i], lo, hi)
            normed[i] = (x_i - lo) / (hi - lo + 1e-8)
        return normed

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

@torch.no_grad()
def run_inference_LDM_aleatoric_and_log_uncertainty_propagation_sparsification(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
        mc_decode_samples: int = 10,
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
    print(f"num_steps: {num_steps}")
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
        pred_noise, pred_logvar = diffusion_model(
            x=model_input,
            timesteps=t_tensor,
            context=None
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
            var_eps = torch.exp(pred_logvar.float())

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
        x, _ = scheduler.step(pred_noise, t_tensor, x)

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
    # condition_dec = autoencoder.decode(condition_batch / scaling)

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
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()

    unc_raw = uncertainty_map.cpu().detach()
    err_raw = abs(pred - gt)

    u = unc_raw.flatten()
    e = err_raw.flatten()

    fractions, curve = sparsification_curve(u, e, max_frac=0.95)
    rand_curve = random_sparsification(e, fractions)
    _, curve_oracle = sparsification_curve(e, e, max_frac=0.95)

    for f, c, r, o in zip(fractions, curve, rand_curve, curve_oracle):
        csv_writer.writerow({
            'Sample': step * B + i,
            'Fraction': f,
            'Error': c,
            'RandomError': r,
            'OracleError': o,
        })

####################### LDM vanilla #####################

@torch.no_grad()
def run_inference_LDM_vanilla_and_log(
        diffusion_model,
        autoencoder,
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
    autoencoder.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        pred_noise = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(pred_noise, t_tensor, x)

    pred_denoised_latent = x

    pred_denoised = autoencoder.decode(pred_denoised_latent/scaling)
    condition_batch = autoencoder.decode(condition_batch/scaling)

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

    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    error = norm_percentile(abs(pred - gt))

    fig, axes = plt.subplots(nrows=B, ncols=5, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Error"]

        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        mask = gt_array != 0

        # Compute metrics
        psnr = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array[mask] - pred_array[mask]) ** 2)
        print(psnr, ssim, mse)
        csv_writer.writerow({'Sample': step * B + i, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim})

        for j in range(4):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()


@torch.no_grad()
def run_inference_LDM_vanilla_and_log_MC_sampling(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
):
    diffusion_model.eval()
    autoencoder.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    # --------------------------------------------------
    # Monte Carlo sampling parameters
    # --------------------------------------------------
    S = 10  # number of sampled trajectories (8–16 is standard)

    samples = []

    # --------------------------------------------------
    # Monte Carlo sampling
    # --------------------------------------------------
    for s in range(S):
        x = torch.randn_like(condition_batch)

        for t in scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device).long()
            model_input = torch.cat([x, condition_batch], dim=1)

            pred_noise = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=None
            )

            x, _ = scheduler.step(pred_noise, t_tensor, x)

        # pred_denoised = x
        x = autoencoder.decode(x / scaling)
        samples.append(x.cpu())

    samples = torch.stack(samples, dim=0)  # (S, B, C, H, W)

    # --------------------------------------------------
    # Predictive mean and sampling variance
    # --------------------------------------------------
    pred_denoised = samples.mean(dim=0)
    mc_uncertainty_map = samples.var(dim=0, unbiased=False)

    # --------------------------------------------------
    # Normalization helper
    # --------------------------------------------------
    def norm_percentile(x, pmin=1, pmax=99):
        x = x.clone().to(torch.float32)
        B = x.shape[0]
        normed = torch.zeros_like(x)
        for i in range(B):
            x_i = x[i]
            lo = torch.quantile(x_i, pmin / 100.0)
            hi = torch.quantile(x_i, pmax / 100.0)
            x_i = torch.clamp(x_i, lo, hi)
            normed[i] = (x_i - lo) / (hi - lo + 1e-8)
        return normed

    # --------------------------------------------------
    # Metrics & logging
    # --------------------------------------------------
    condition_batch = autoencoder.decode(condition_batch / scaling)
    ld = condition_batch.cpu()
    gt = gt_batch.cpu()
    pred = pred_denoised.cpu()
    unc = norm_percentile(mc_uncertainty_map).cpu()
    error = norm_percentile(torch.abs(pred - gt))

    fig, axes = plt.subplots(nrows=B, ncols=5, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "MC-Dropout Unc.", "Error"]

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
        correlations_norm = map_correlations_multi_thresholds(unc[i][0].cpu().detach().numpy(), pred_array, gt_array)
        correlations_unnorm = map_correlations_multi_thresholds(mc_uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)

        csv_writer.writerow({'Sample': step * B + i,
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
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["MC-Dropout Unc.", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference_MC_Dropout", plt.gcf(), global_step=step)
    plt.close()

@torch.no_grad()
def run_inference_LDM_vanilla_and_log_MC_sampling_uncertainty_eval(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
        n_sampling,
):
    diffusion_model.eval()
    autoencoder.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    # --------------------------------------------------
    # Monte Carlo sampling parameters
    # --------------------------------------------------
    S = n_sampling  # number of sampled trajectories (8–16 is standard)

    samples = []

    # --------------------------------------------------
    # Monte Carlo sampling
    # --------------------------------------------------
    for s in range(S):
        x = torch.randn_like(condition_batch)

        for t in scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device).long()
            model_input = torch.cat([x, condition_batch], dim=1)

            pred_noise = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=None
            )

            x, _ = scheduler.step(pred_noise, t_tensor, x)

        # pred_denoised = x
        x = autoencoder.decode(x / scaling)
        samples.append(x.cpu())

    samples = torch.stack(samples, dim=0)  # (S, B, C, H, W)

    # --------------------------------------------------
    # Predictive mean and sampling variance
    # --------------------------------------------------
    pred_denoised = samples.mean(dim=0)
    mc_uncertainty_map = samples.var(dim=0, unbiased=False)

    # --------------------------------------------------
    # Metrics & logging
    # --------------------------------------------------
    gt = gt_batch.cpu()
    pred = pred_denoised.cpu()


    for i in range(B):
        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        mask = gt_array != -1

        # --- Calibration data ---
        unc_raw = mc_uncertainty_map[i][0].cpu()

        ######### Compute metrics #########

        mae = np.mean(np.abs(gt_array[mask] - pred_array[mask]))
        uncertainty_summary = summarize_uncertainty(unc_raw[mask])

        csv_writer.writerow({'Sample': step * B + i,
                             'MAE': mae,
                             'u_mean': uncertainty_summary["u_mean"],
                             'u_p95': uncertainty_summary["u_p95"],
                             'u_p99': uncertainty_summary["u_p99"],
                             'u_top1_mean': uncertainty_summary["u_top1_mean"],
                             'top5_u_mean': uncertainty_summary["u_top5_mean"],
                             })

    plt.close()

@torch.no_grad()
def run_inference_LDM_vanilla_and_log_MC_sampling_sparsification(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
        n_sampling,
):
    diffusion_model.eval()
    autoencoder.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    # --------------------------------------------------
    # Monte Carlo sampling parameters
    # --------------------------------------------------
    S = n_sampling  # number of sampled trajectories (8–16 is standard)

    samples = []

    # --------------------------------------------------
    # Monte Carlo sampling
    # --------------------------------------------------
    for s in range(S):
        x = torch.randn_like(condition_batch)

        for t in scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device).long()
            model_input = torch.cat([x, condition_batch], dim=1)

            pred_noise = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=None
            )

            x, _ = scheduler.step(pred_noise, t_tensor, x)

        # pred_denoised = x
        x = autoencoder.decode(x / scaling)
        samples.append(x.cpu())

    samples = torch.stack(samples, dim=0)  # (S, B, C, H, W)

    # --------------------------------------------------
    # Predictive mean and sampling variance
    # --------------------------------------------------
    pred_denoised = samples[0]
    mc_uncertainty_map = samples.var(dim=0, unbiased=False)


    # --------------------------------------------------
    # Metrics & logging
    # --------------------------------------------------

    gt = gt_batch.cpu()
    pred = pred_denoised.cpu()
    unc_raw = mc_uncertainty_map.cpu().detach()
    err_raw = abs(pred - gt)

    u = unc_raw.flatten()
    e = err_raw.flatten()

    fractions, curve = sparsification_curve_fast(u, e, max_frac=0.95)
    rand_curve = random_sparsification_fast(e, fractions)
    _, curve_oracle = sparsification_curve_fast(e, e, max_frac=0.95)

    for f, c, r, o in zip(fractions, curve, rand_curve, curve_oracle):
        csv_writer.writerow({
            'Sample': step * B,
            'Fraction': f,
            'Error': c,
            'RandomError': r,
            'OracleError': o,
        })