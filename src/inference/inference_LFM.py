from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
import torch
import os
from src.inference.utils import sparsification_curve, random_sparsification, norm_percentile, collect_calibration_data, map_correlations_multi_thresholds


####################### Latent Rectified Flow Matching self-refining #########################
@torch.no_grad()
def run_inference_LFM_self_refining_and_log_uncertainty_propagation(
        diffusion_model,
        autoencoder,
        context_encoder,
        writer,
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
    U_v = torch.zeros_like(x)
    num_valid_steps = 0

    # ==================================================
    # DDIM sampling in latent space
    #
    # We follow a standard deterministic DDIM trajectory.
    # Uncertainty is accumulated but NEVER injected into x.
    # ==================================================
    all_next_timesteps = torch.cat((scheduler.timesteps[1:],torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total = min(len(scheduler.timesteps), len(all_next_timesteps)),)):
        t_tensor = torch.tensor([t], device=device).long()

        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        # ==================================================
        # (i == 0): double forward
        # ==================================================
        if i == 0:
            # ---- pass 1: dummy context
            with autocast(True):
                dummy_context = torch.zeros((1, 1, 128), device=device)
                predicted_velocity_1, pred_logvar_1 = diffusion_model(x=model_input, timesteps=t_tensor, context=dummy_context)

            # ---- build uncertainty
            uncertainty_map = norm_percentile(
                torch.exp(pred_logvar_1.float())
            )

            context_vector = context_encoder(uncertainty_map)

            # ---- pass 2: refined
            with autocast(True):
                predicted_velocity, pred_logvar = diffusion_model(
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
                predicted_velocity, pred_logvar = diffusion_model(
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
        # ==================================================
        # Accumulate late-step decision-time uncertainty
        # ==================================================
        if (num_steps - K - 1) <= i < (num_steps - 1):

            # Learned variance of the velocity field
            var_v_t = torch.exp(pred_logvar.float())

            # Optional: weight by dt^2 (commented out by default)
            # dt = 1.0 / scheduler.num_inference_steps
            # var_v_t = (dt ** 2) * var_v_t

            U_v += var_v_t
            num_valid_steps += 1

        # ==================================================
        # Deterministic DDIM update
        #
        # Note: uncertainty does NOT affect the trajectory.
        # ==================================================
        x, _ = scheduler.step(predicted_velocity,  t, x, next_t)

    # ======================================================
    # Final latent mean and aggregated uncertainty
    #
    # pred_denoised_latent is the mean prediction μ_z0
    # ======================================================
    pred_denoised_latent = x

    # --------------------------------------------------
    # Average accumulated variance across K steps
    # --------------------------------------------------
    var_v = U_v / max(num_valid_steps, 1)

    # --------------------------------------------------
    # Convert variance to standard deviation
    #
    # Required because we will SAMPLE latent perturbations.
    # --------------------------------------------------
    sigma_z0 = torch.sqrt(var_v.clamp_min(1e-12))

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
        z0_s = pred_denoised_latent + sigma_z0 *eps
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
def run_inference_LFM_self_refining_and_log_uncertainty_propagation_sparsification(
        diffusion_model,
        autoencoder,
        context_encoder,
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
    U_v = torch.zeros_like(x)
    num_valid_steps = 0

    # ==================================================
    # DDIM sampling in latent space
    #
    # We follow a standard deterministic DDIM trajectory.
    # Uncertainty is accumulated but NEVER injected into x.
    # ==================================================
    all_next_timesteps = torch.cat((scheduler.timesteps[1:],torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total = min(len(scheduler.timesteps), len(all_next_timesteps)),)):
        t_tensor = torch.tensor([t], device=device).long()

        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        # ==================================================
        # (i == 0): double forward
        # ==================================================
        if i == 0:
            # ---- pass 1: dummy context
            with autocast(True):
                dummy_context = torch.zeros((1, 1, 128), device=device)
                predicted_velocity_1, pred_logvar_1 = diffusion_model(x=model_input, timesteps=t_tensor, context=dummy_context)

            # ---- build uncertainty
            uncertainty_map = norm_percentile(
                torch.exp(pred_logvar_1.float())
            )

            context_vector = context_encoder(uncertainty_map)

            # ---- pass 2: refined
            with autocast(True):
                predicted_velocity, pred_logvar = diffusion_model(
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
                predicted_velocity, pred_logvar = diffusion_model(
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
        # ==================================================
        # Accumulate late-step decision-time uncertainty
        # ==================================================
        if (num_steps - K - 1) <= i < (num_steps - 1):

            # Learned variance of the velocity field
            var_v_t = torch.exp(pred_logvar.float())

            # Optional: weight by dt^2 (commented out by default)
            # dt = 1.0 / scheduler.num_inference_steps
            # var_v_t = (dt ** 2) * var_v_t

            U_v += var_v_t
            num_valid_steps += 1

        # ==================================================
        # Deterministic DDIM update
        #
        # Note: uncertainty does NOT affect the trajectory.
        # ==================================================
        x, _ = scheduler.step(predicted_velocity,  t, x, next_t)

    # ======================================================
    # Final latent mean and aggregated uncertainty
    #
    # pred_denoised_latent is the mean prediction μ_z0
    # ======================================================
    pred_denoised_latent = x

    # --------------------------------------------------
    # Average accumulated variance across K steps
    # --------------------------------------------------
    var_v = U_v / max(num_valid_steps, 1)

    # --------------------------------------------------
    # Convert variance to standard deviation
    #
    # Required because we will SAMPLE latent perturbations.
    # --------------------------------------------------
    sigma_z0 = torch.sqrt(var_v.clamp_min(1e-12))

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
            'Sample': step * B,
            'Fraction': f,
            'Error': c,
            'RandomError': r,
            'OracleError': o,
        })

############################ LFM vanilla ##################################

@torch.no_grad()
def run_inference_LFM_vanilla_and_log(
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

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total=min(len(scheduler.timesteps), len(all_next_timesteps)), )):
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        predicted_velocity = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(predicted_velocity, t, x, next_t)

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
def run_inference_LFM_vanilla_and_log_MC_sampling(
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
    B, C, H, W = condition_batch.shape


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
        all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
        for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total=min(len(scheduler.timesteps), len(all_next_timesteps)), )):
            t_tensor = torch.tensor([t], device=device).long()
            model_input = torch.cat([x, condition_batch], dim=1)

            predicted_velocity = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=None
            )

            x, _ = scheduler.step(predicted_velocity,  t, x, next_t)

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
def run_inference_LFM_vanilla_and_log_MC_sampling_sparsification(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        step,
        device,
        scheduler,
        scaling,
        csv_writer,
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape


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
        all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
        for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total=min(len(scheduler.timesteps), len(all_next_timesteps)), )):
            t_tensor = torch.tensor([t], device=device).long()
            model_input = torch.cat([x, condition_batch], dim=1)

            predicted_velocity = diffusion_model(
                x=model_input,
                timesteps=t_tensor,
                context=None
            )

            x, _ = scheduler.step(predicted_velocity,  t, x, next_t)

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

    unc_raw = mc_uncertainty_map.cpu().detach()
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

######################### Latent Rectified Flow Matching aleatoric #########################
@torch.no_grad()
def run_inference_LFM_aleatoric_and_log_uncertainty_propagation(
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
        K: int = 10,   # number of late steps used for uncertainty aggregation
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
    U_v = torch.zeros_like(x)
    num_valid_steps = 0

    # ==================================================
    # DDIM sampling in latent space
    #
    # We follow a standard deterministic DDIM trajectory.
    # Uncertainty is accumulated but NEVER injected into x.
    # ==================================================
    all_next_timesteps = torch.cat((scheduler.timesteps[1:],torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total = min(len(scheduler.timesteps), len(all_next_timesteps)),)):
        t_tensor = torch.tensor([t], device=device).long()

        model_input = torch.cat([x, condition_batch], dim=1)

        predicted_velocity, pred_logvar = diffusion_model(
            x=model_input,
            timesteps=t_tensor,
            context=None,
        )

        # ==================================================
        # Accumulate late-step decision-time uncertainty
        # ==================================================
        if (num_steps - K - 1) <= i < (num_steps - 1):
            # Learned variance of the velocity field
            var_v_t = torch.exp(pred_logvar.float())

            # Optional: weight by dt^2 (commented out by default)
            # dt = 1.0 / scheduler.num_inference_steps
            # var_v_t = (dt ** 2) * var_v_t

            U_v += var_v_t
            num_valid_steps += 1

        # ==================================================
        # Deterministic DDIM update
        #
        # Note: uncertainty does NOT affect the trajectory.
        # ==================================================
        x, _ = scheduler.step(predicted_velocity, t, x, next_t)

    # ======================================================
    # Final latent mean and aggregated uncertainty
    #
    # pred_denoised_latent is the mean prediction μ_z0
    # ======================================================
    pred_denoised_latent = x

    # --------------------------------------------------
    # Average accumulated variance across K steps
    # --------------------------------------------------
    var_v = U_v / max(num_valid_steps, 1)

    # --------------------------------------------------
    # Convert variance to standard deviation
    #
    # Required because we will SAMPLE latent perturbations.
    # --------------------------------------------------
    sigma_z0 = torch.sqrt(var_v.clamp_min(1e-12))

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
def run_inference_LFM_aleatoric_and_log_uncertainty_propagation_sparsification(
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
        K: int = 10,   # number of late steps used for uncertainty aggregation
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
    U_v = torch.zeros_like(x)
    num_valid_steps = 0

    # ==================================================
    # DDIM sampling in latent space
    #
    # We follow a standard deterministic DDIM trajectory.
    # Uncertainty is accumulated but NEVER injected into x.
    # ==================================================
    all_next_timesteps = torch.cat((scheduler.timesteps[1:],torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for i, (t, next_t) in enumerate(tqdm(zip(scheduler.timesteps, all_next_timesteps), total = min(len(scheduler.timesteps), len(all_next_timesteps)),)):
        t_tensor = torch.tensor([t], device=device).long()

        model_input = torch.cat([x, condition_batch], dim=1)

        predicted_velocity, pred_logvar = diffusion_model(
            x=model_input,
            timesteps=t_tensor,
            context=None,
        )

        # ==================================================
        # Accumulate late-step decision-time uncertainty
        # ==================================================
        if (num_steps - K - 1) <= i < (num_steps - 1):
            # Learned variance of the velocity field
            var_v_t = torch.exp(pred_logvar.float())

            # Optional: weight by dt^2 (commented out by default)
            # dt = 1.0 / scheduler.num_inference_steps
            # var_v_t = (dt ** 2) * var_v_t

            U_v += var_v_t
            num_valid_steps += 1

        # ==================================================
        # Deterministic DDIM update
        #
        # Note: uncertainty does NOT affect the trajectory.
        # ==================================================
        x, _ = scheduler.step(predicted_velocity, t, x, next_t)

    # ======================================================
    # Final latent mean and aggregated uncertainty
    #
    # pred_denoised_latent is the mean prediction μ_z0
    # ======================================================
    pred_denoised_latent = x

    # --------------------------------------------------
    # Average accumulated variance across K steps
    # --------------------------------------------------
    var_v = U_v / max(num_valid_steps, 1)

    # --------------------------------------------------
    # Convert variance to standard deviation
    #
    # Required because we will SAMPLE latent perturbations.
    # --------------------------------------------------
    sigma_z0 = torch.sqrt(var_v.clamp_min(1e-12))

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
