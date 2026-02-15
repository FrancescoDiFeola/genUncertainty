from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
import torch
import os
from src.inference.utils import sparsification_curve, random_sparsification, norm_percentile, collect_calibration_data, map_correlations_multi_thresholds

############### DDPM self-refining ####################
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
    # sigma_x0 = torch.sqrt(torch.clamp(sigma2_x0, min=1e-12))

    return sigma2_x0


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
def run_inference_and_log_v2(  # in this function the uncertainty is used for iterative refinement with a two-forward strategy
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

        # --------------------------------------
        # 🔁 Second Pass: conditioned refinement
        # --------------------------------------
        with autocast(enabled=True):
            pred_error_2, pred_logvar_2 = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=context_vector
            )

        # Second-pass uncertainty (store + save PNG)
        second_pass_uncertainty = torch.exp(pred_logvar_2).detach()
        norm_second_pass = norm_percentile(second_pass_uncertainty)  # .cpu()  # (B,1,H,W)
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
def run_inference_and_log_v3(  # in this function the uncertainty is used for iterative refinement without two-forward
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
        csv_writer,
        csv_writer_2,
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

        # ----------------------------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------------------------
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

        # -------------------------------------
        # 🔁 Second Pass: conditioned refinement
        # --------------------------------------
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
    # uncertainty_map = torch.exp(uncertainty)

    last_t = t_tensor
    # --- Option A: analytic uncertainty propagation ---
    uncertainty_map = propagate_uncertainty_eps_to_x0(
        pred_logvar_eps=uncertainty,
        timesteps=last_t,
        scheduler=scheduler,
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
        # mask = gt_array != 0

        # --- Calibration data ---
        unc_raw = uncertainty_map[i][0].cpu()
        err_raw = (pred[i][0] - gt[i][0]).abs()

        bin_u, bin_e, bin_n = collect_calibration_data(
            unc_raw,
            err_raw,
            num_bins=15
        )

        # Log per-bin values for later plotting
        for k in range(len(bin_u)):
            csv_writer_2.writerow({
                'Sample': step * B + i,
                'Bin': k,
                'Unc_mean': bin_u[k],
                'Err_mean': bin_e[k],
                'Count': bin_n[k],
                'Type': 'calibration'
            })

        ######### Compute metrics #########
        psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        print(gt_array.shape, pred_array.shape, unc[i][0].shape)
        ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array - pred_array) ** 2)
        # psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # mse = np.mean((gt_array - pred_array) ** 2)
        correlations_norm = map_correlations_multi_thresholds(unc[i][0].cpu().detach().numpy(), pred_array, gt_array)
        correlations_unnorm = map_correlations_multi_thresholds(uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)

        print(psnr, ssim, mse,
              correlations_norm["pearson"],
              correlations_norm["spearman"],
              correlations_norm["AUROC_top15"],
              correlations_norm["AUROC_top10"],
              correlations_norm["AUROC_top5"],
              correlations_unnorm["pearson"],
              correlations_unnorm["spearman"],
              correlations_unnorm["AUROC_top15"],
              correlations_unnorm["AUROC_top10"],
              correlations_unnorm["AUROC_top5"])

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
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()


@torch.no_grad()
def run_inference_and_log_v3_clean(  # in this function the uncertainty is used for iterative refinement without two-forward
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
        csv_writer,
        csv_writer_2,
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

        model_input = torch.cat([x, condition_batch], dim=1)
        # ==================================================
        # (i == 0): double forward
        # ==================================================
        if i == 0:
            # ---- pass 1: dummy context
            with autocast(True):
                dummy_context = torch.zeros((1, 1, 128), device=device)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=dummy_context
                )

            # ---- build uncertainty
            uncertainty_map = norm_percentile(
                torch.exp(pred_logvar_1.float())
            )

            context_vector = context_encoder(uncertainty_map)

            # ---- pass 2: refined
            with autocast(True):
                pred_error, pred_logvar = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )

        # ==================================================
        # ALL FOLLOWING STEPS: single forward
        # ==================================================
        else:
            context_vector = context_encoder(prev_uncertainty_map)

            with autocast(True):
                pred_error, pred_logvar = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )
        # ==================================================
        # Update uncertainty memory
        # ==================================================
        prev_uncertainty_map = norm_percentile(
            torch.exp(pred_logvar.float())
        )

        # ==================================================
        # DDIM update
        # ==================================================
        x, _ = scheduler.step(pred_error, t_tensor, x)

        norm_second = norm_percentile(prev_uncertainty_map).cpu()

        # Save each sample's uncertainty as PNG
        if int(step) == 1:
            for b in range(B):
                arr = norm_second[b].squeeze(0).numpy()
                png_path = os.path.join(dir, f"sample{b}_{step}_t_step{int(t)}_epoch_350.png")
                plt.imsave(png_path, arr, cmap='hot')

    pred_denoised = x
    # uncertainty_map = torch.exp(uncertainty)

    last_t = t_tensor

    # --- Option A: analytic uncertainty propagation ---
    uncertainty_map = propagate_uncertainty_eps_to_x0(
        pred_logvar_eps=pred_logvar.float(),
        timesteps=last_t,
        scheduler=scheduler,
    )

    print(uncertainty_map.mean(), torch.exp(pred_logvar.float()).mean())

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
        # mask = gt_array != 0

        # --- Calibration data ---
        unc_raw = uncertainty_map[i][0].cpu()
        err_raw = (pred[i][0] - gt[i][0]).abs()

        bin_u, bin_e, bin_n = collect_calibration_data(
            unc_raw,
            err_raw,
            num_bins=15
        )

        # Log per-bin values for later plotting
        for k in range(len(bin_u)):
            csv_writer_2.writerow({
                'Sample': step * B + i,
                'Bin': k,
                'Unc_mean': bin_u[k],
                'Err_mean': bin_e[k],
                'Count': bin_n[k],
                'Type': 'calibration'
            })

        ######### Compute metrics #########
        psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array - pred_array) ** 2)
        # psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # mse = np.mean((gt_array - pred_array) ** 2)
        correlations_norm = map_correlations_multi_thresholds(unc[i][0].cpu().detach().numpy(), pred_array, gt_array)
        correlations_unnorm = map_correlations_multi_thresholds(uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)

        print(psnr, ssim, mse,
              correlations_norm["pearson"],
              correlations_norm["spearman"],
              correlations_norm["AUROC_top15"],
              correlations_norm["AUROC_top10"],
              correlations_norm["AUROC_top5"],
              correlations_unnorm["pearson"],
              correlations_unnorm["spearman"],
              correlations_unnorm["AUROC_top15"],
              correlations_unnorm["AUROC_top10"],
              correlations_unnorm["AUROC_top5"])

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
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()


@torch.no_grad()
def run_inference_and_log_v3_clean_unc_integral(  # in this function the uncertainty is used for iterative refinement without two-forward
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
        csv_writer,
        csv_writer_2,
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    # ==================================================
    # Trajectory-level uncertainty accumulator (x0 space)
    # ==================================================
    U_x0 = torch.zeros((B, 1, H, W), device=device)
    num_valid_steps = 0

    num_steps = len(scheduler.timesteps)
    K = 10  # number of late decision-relevant steps (excluding final)
    prev_uncertainty_map = None
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM Sampling")):

        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)

        # ==================================================
        # (i == 0): double forward
        # ==================================================
        if i == 0:
            # ---- pass 1: dummy context
            with autocast(True):
                dummy_context = torch.zeros((1, 1, 128), device=device)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=dummy_context
                )

            # ---- build uncertainty
            uncertainty_map = norm_percentile(
                torch.exp(pred_logvar_1.float())
            )

            context_vector = context_encoder(uncertainty_map)

            # ---- pass 2: refined
            with autocast(True):
                pred_error, pred_logvar = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )

        # ==================================================
        # ALL FOLLOWING STEPS: single forward
        # ==================================================
        else:
            context_vector = context_encoder(prev_uncertainty_map)

            with autocast(True):
                pred_error, pred_logvar = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )
        # ==================================================
        # Update uncertainty memory
        # ==================================================
        prev_uncertainty_map = norm_percentile(
            torch.exp(pred_logvar.float())
        )

        # ==================================================
        # Accumulate late-step decision-time uncertainty (evaluation only)
        # ==================================================
        # Use last K steps, excluding the final step
        if (num_steps - K - 1) <= i < (num_steps - 1):
            a_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)

            var_eps = torch.exp(pred_logvar.float())
            var_x0_t = (1.0 - a_bar) / (a_bar + 1e-8) * var_eps

            U_x0 += var_x0_t
            num_valid_steps += 1

        # ==================================================
        # DDIM update
        # ==================================================
        x, _ = scheduler.step(pred_error, t_tensor, x)

        norm_second = norm_percentile(prev_uncertainty_map).cpu()

        # Save each sample's uncertainty as PNG
        if int(step) == 1:
            for b in range(B):
                arr = norm_second[b].squeeze(0).numpy()
                png_path = os.path.join(dir, f"sample{b}_{step}_t_step{int(t)}_epoch_350.png")
                plt.imsave(png_path, arr, cmap='hot')

    pred_denoised = x
    # uncertainty_map = torch.exp(uncertainty)

    # Final trajectory-integrated uncertainty map (for metrics only)
    U_x0 = U_x0 / max(num_valid_steps, 1)
    uncertainty_map = U_x0

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
        # mask = gt_array != 0

        # --- Calibration data ---
        unc_raw = uncertainty_map[i][0].cpu()
        err_raw = (pred[i][0] - gt[i][0]).abs()

        bin_u, bin_e, bin_n = collect_calibration_data(
            unc_raw,
            err_raw,
            num_bins=15
        )

        # Log per-bin values for later plotting
        for k in range(len(bin_u)):
            csv_writer_2.writerow({
                'Sample': step * B + i,
                'Bin': k,
                'Unc_mean': bin_u[k],
                'Err_mean': bin_e[k],
                'Count': bin_n[k],
                'Type': 'calibration'
            })

        ######### Compute metrics #########
        psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array - pred_array) ** 2)
        # psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        # mse = np.mean((gt_array - pred_array) ** 2)
        correlations_norm = map_correlations_multi_thresholds(unc[i][0].cpu().detach().numpy(), pred_array, gt_array)
        correlations_unnorm = map_correlations_multi_thresholds(uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)

        print(psnr, ssim, mse,
              correlations_norm["pearson"],
              correlations_norm["spearman"],
              correlations_norm["AUROC_top15"],
              correlations_norm["AUROC_top10"],
              correlations_norm["AUROC_top5"],
              correlations_unnorm["pearson"],
              correlations_unnorm["spearman"],
              correlations_unnorm["AUROC_top15"],
              correlations_unnorm["AUROC_top10"],
              correlations_unnorm["AUROC_top5"])

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
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()

@torch.no_grad()
def run_inference_and_log_v3_clean_unc_integral_sparsification(
        diffusion_model,
        context_encoder,
        dir,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        csv_writer,
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(50)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    # ==================================================
    # Trajectory-level uncertainty accumulator (x0 space)
    # ==================================================
    U_x0 = torch.zeros((B, 1, H, W), device=device)
    num_valid_steps = 0

    num_steps = len(scheduler.timesteps)
    K = 10  # number of late decision-relevant steps (excluding final)
    prev_uncertainty_map = None
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM Sampling")):

        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)

        # ==================================================
        # (i == 0): double forward
        # ==================================================
        if i == 0:
            # ---- pass 1: dummy context
            with autocast(True):
                dummy_context = torch.zeros((1, 1, 128), device=device)
                pred_error_1, pred_logvar_1 = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=dummy_context
                )

            # ---- build uncertainty
            uncertainty_map = norm_percentile(
                torch.exp(pred_logvar_1.float())
            )

            context_vector = context_encoder(uncertainty_map)

            # ---- pass 2: refined
            with autocast(True):
                pred_error, pred_logvar = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )

        # ==================================================
        # ALL FOLLOWING STEPS: single forward
        # ==================================================
        else:
            context_vector = context_encoder(prev_uncertainty_map)

            with autocast(True):
                pred_error, pred_logvar = diffusion_model(
                    x=model_input,
                    timesteps=t_tensor,
                    context=context_vector
                )
        # ==================================================
        # Update uncertainty memory
        # ==================================================
        prev_uncertainty_map = norm_percentile(
            torch.exp(pred_logvar.float())
        )

        # ==================================================
        # Accumulate late-step decision-time uncertainty (evaluation only)
        # ==================================================
        # Use last K steps, excluding the final step
        if (num_steps - K - 1) <= i < (num_steps - 1):
            a_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)

            var_eps = torch.exp(pred_logvar.float())
            var_x0_t = (1.0 - a_bar) / (a_bar + 1e-8) * var_eps

            U_x0 += var_x0_t
            num_valid_steps += 1

        # ==================================================
        # DDIM update
        # ==================================================
        x, _ = scheduler.step(pred_error, t_tensor, x)

        norm_second = norm_percentile(prev_uncertainty_map).cpu()

        # Save each sample's uncertainty as PNG
        if int(step) == 1:
            for b in range(B):
                arr = norm_second[b].squeeze(0).numpy()
                png_path = os.path.join(dir, f"sample{b}_{step}_t_step{int(t)}_epoch_350.png")
                plt.imsave(png_path, arr, cmap='hot')

    pred_denoised = x
    # uncertainty_map = torch.exp(uncertainty)

    # Final trajectory-integrated uncertainty map (for metrics only)
    U_x0 = U_x0 / max(num_valid_steps, 1)
    uncertainty_map = U_x0

    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()

    unc_raw =uncertainty_map.cpu().detach()
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

############### DDPM vanilla ################
@torch.no_grad()
def run_ddpm_vanilla_inference_and_log(
        diffusion_model,
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

    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        pred_noise = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(pred_noise, t_tensor, x)


    pred_denoised = x

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
def run_ddpm_vanilla_inference_and_log_MC_sampling(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        csv_writer,
        csv_writer_2,
):
    diffusion_model.eval()
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


        # --- Calibration data ---
        unc_raw = mc_uncertainty_map[i][0].cpu()
        err_raw = (pred[i][0] - gt[i][0]).abs()

        bin_u, bin_e, bin_n = collect_calibration_data(
            unc_raw,
            err_raw,
            num_bins=15
        )

        # Log per-bin values for later plotting
        for k in range(len(bin_u)):
            csv_writer_2.writerow({
                'Sample': step * B + i,
                'Bin': k,
                'Unc_mean': bin_u[k],
                'Err_mean': bin_e[k],
                'Count': bin_n[k],
                'Type': 'calibration'
            })


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
def run_ddpm_vanilla_inference_and_log_MC_sampling_sparsification(
        diffusion_model,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        csv_writer,
):
    diffusion_model.eval()
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
    ld = condition_batch.cpu()
    gt = gt_batch.cpu()
    pred = pred_denoised.cpu()

    unc_raw =mc_uncertainty_map.cpu().detach()
    err_raw = abs(pred - gt)

    u = unc_raw.flatten()
    e = err_raw.flatten()


    fractions, curve = sparsification_curve(u, e, max_frac=0.95)
    rand_curve = random_sparsification(e, fractions)
    _, curve_oracle = sparsification_curve(e, e, max_frac=0.95)

    for f, c, r, o in zip(fractions, curve, rand_curve, curve_oracle):
        csv_writer.writerow({
            'Sample': step * B,
            'Fraction': f,
            'Error': c,
            'RandomError': r,
            'OracleError': o,
        })
