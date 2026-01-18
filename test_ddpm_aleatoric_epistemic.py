import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from src import T1T2Dataset
from src import LDCTHDCTDataset
from src import networks
from src import FixedMaskDropout
# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


@torch.no_grad()
def sample_and_plot_batch_ddim_(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        csv_writer,
        tag="DDIM_Sampling",
        scheduler=DDIMScheduler,
        num_training_steps=1000,
        num_inference_steps=50,
        beta_start=0.0015,
        beta_end=0.0205,
        epistemic_samples=5,
):
    """
    DDIM sampling + tensorboard batch display with uncertainty.
    Plots [LD | GT | Prediction | Aleatoric | Epistemic | Total | Error] per row.
    """

    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(num_inference_steps)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    progress = tqdm(scheduler.timesteps, desc="DDIM Sampling")

    for t in progress:
        t_tensor = torch.tensor([t], device=device).long()

        # Reconstruct input with current latent
        model_input = torch.cat([x, condition_batch], dim=1)

        h = diffusion.module.forward_backbone(x=model_input, timesteps=t_tensor, context=None) if NUM_GPUS > 1 else diffusion.forward_backbone(x=model_input, timesteps=t_tensor, context=None)

        # Run aleatoric head once
        pred_logvar = diffusion.module.forward_logvar_head(h) if NUM_GPUS > 1 else diffusion.forward_logvar_head(h)

        # run multiple
        pred_means = []
        for _ in range(epistemic_samples):
            pred_mean = diffusion.module.forward_mean_head(h) if NUM_GPUS > 1 else diffusion.forward_mean_head(h)
            pred_means.append(pred_mean.unsqueeze(0))


        pred_noises = torch.cat(pred_means, dim=0)
        pred_noise_mean = pred_noises.mean(dim=0)
        epistemic_var = pred_noises.var(dim=0)


        # Use logvar from last sample (they should be stable across samples)
        aleatoric_var = torch.exp(pred_logvar)
        total_var = aleatoric_var + epistemic_var
        # logvar_total = torch.log(total_var + 1e-8)

        # Step in scheduler
        x, _ = scheduler.step(pred_noise_mean, t_tensor, x)

    pred_denoised = x


    # Normalize outputs
    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x

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
    aleatoric = norm_percentile(aleatoric_var).cpu().detach()
    epistemic = norm_percentile(epistemic_var).cpu().detach()
    total = norm_percentile(total_var).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    # Create figure
    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=7, figsize=(12, 2.5 * num_samples))
    if B == 1:
        axes = [axes]  # make iterable

    for i in range(num_samples):
        images = [ld[i], gt[i], pred[i], aleatoric[i], epistemic[i], total[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Aleatoric", "Epistemic", "Total", "Error"]
        
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


        for j in range(7):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()

            if titles[j] in ["Aleatoric", "Epistemic", "Total", "Error"]:
                ax.imshow(img, cmap='hot')
            else:
                ax.imshow(img, cmap='gray')

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.close()


@torch.no_grad()
def sample_and_plot_batch_ddim(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        csv_writer,
        tag="DDIM_Sampling",
        scheduler=DDIMScheduler,
        num_training_steps=1000,
        num_inference_steps=50,
        beta_start=0.0015,
        beta_end=0.0205,
        epistemic_samples=5,
):
    """
    DDIM sampling + tensorboard batch display with uncertainty.
    Plots [LD | GT | Prediction | Aleatoric | Epistemic | Total | Error] per row.
    """

    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    scheduler.set_timesteps(num_inference_steps)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    progress = tqdm(scheduler.timesteps, desc="DDIM Sampling")

    all_preds = []

    for epi in range(epistemic_samples):
        # ✅ Reset dropout mask at the start of this trajectory
        for m in diffusion.modules():
            if isinstance(m, FixedMaskDropout):
                m.reset_mask()

        # Start new trajectory
        x = torch.randn_like(condition_batch).to(device)

        for t in progress:
            t_tensor = torch.tensor([t], device=device).long()

            # Reconstruct input with current latent
            model_input = torch.cat([x, condition_batch], dim=1)

            # Backbone
            h = (
                diffusion.module.forward_backbone(x=model_input, timesteps=t_tensor, context=None)
                if NUM_GPUS > 1 else
                diffusion.forward_backbone(x=model_input, timesteps=t_tensor, context=None)
            )

            # Aleatoric head (only needed from one trajectory)
            if epi == 0:
                pred_logvar = (
                    diffusion.module.forward_logvar_head(h)
                    if NUM_GPUS > 1 else
                    diffusion.forward_logvar_head(h)
                )

            # Mean head with fixed dropout
            pred_mean = (
                diffusion.module.forward_mean_head(h)
                if NUM_GPUS > 1 else
                diffusion.forward_mean_head(h)
            )

            # Step
            x, _ = scheduler.step(pred_mean, t_tensor, x)

        all_preds.append(x.unsqueeze(0))

    # Stack all final predictions
    pred_stack = torch.cat(all_preds, dim=0)
    pred_denoised = pred_stack.mean(dim=0)
    epistemic_var = pred_stack.var(dim=0)

    # Aleatoric uncertainty from first trajectory
    aleatoric_var = torch.exp(pred_logvar)
    total_var = aleatoric_var + epistemic_var
    
    # Normalize outputs
    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x

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
    aleatoric = norm_percentile(aleatoric_var).cpu().detach()
    epistemic = norm_percentile(epistemic_var).cpu().detach()
    total = norm_percentile(total_var).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    # Create figure
    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=7, figsize=(12, 2.5 * num_samples))
    if B == 1:
        axes = [axes]  # make iterable

    for i in range(num_samples):
        images = [ld[i], gt[i], pred[i], aleatoric[i], epistemic[i], total[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Aleatoric", "Epistemic", "Total", "Error"]

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

        for j in range(7):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()

            if titles[j] in ["Aleatoric", "Epistemic", "Total", "Error"]:
                ax.imshow(img, cmap='hot')
            else:
                ax.imshow(img, cmap='gray')

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=False)
    parser.add_argument('--output_dir', type=str,default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", required=False)
    parser.add_argument('--diff_ckpt', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Checkpoint directory: {experiment_dir}")

    dataset = T1T2Dataset(
        annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
        annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
    )
    
    """
    dataset = LDCTHDCTDataset(
        annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
        annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
    )
    """

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    diffusion = networks.init_ddpm_aleatoric(2, 1, args.diff_ckpt).to(DEVICE)
    print(diffusion)
    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    writer = SummaryWriter(comment=args.experiment_name)
    csv_path = os.path.join(args.output_dir, f"{args.experiment_name}_metrics_epoch_200.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Sample', 'MSE', 'PSNR', 'SSIM']
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

        for step, batch in enumerate(loader):
            sample_and_plot_batch_ddim(
                    diffusion_model=diffusion,
                    condition_batch=batch['A'],
                    gt_batch=batch['B'],
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    csv_writer=writer_csv,
                    tag="DDIM_Sampling",
                    scheduler=scheduler
                )


    print(f"✅ Inference complete. Metrics saved to {csv_path}")
