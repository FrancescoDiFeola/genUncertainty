import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp import networks
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

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
def collect_calibration_data(
    unc_map: torch.Tensor,
    err_map: torch.Tensor,
    num_bins: int = 15,  # number of callibration bins
):
    """
    Collect per-bin statistics for calibration (ECE / reliability).

    Args:
        unc_map: (H, W) uncertainty (std or variance, NOT normalized)
        err_map: (H, W) absolute error
    Returns:
        bin_unc_mean, bin_err_mean, bin_count
    """
    u = unc_map.flatten()
    e = err_map.flatten()

    mask = torch.isfinite(u) & torch.isfinite(e)
    u = u[mask]
    e = e[mask]

    # Define bins over uncertainty  [q1, q2, ...., q15], Each bin contains roughly the same number of pixels, bobust to heavy-tailed uncertainty distributions, Prevents empty bins, standard in the literature
    bins = torch.quantile(u, torch.linspace(0, 1, num_bins + 1, device=u.device))
    bin_ids = torch.bucketize(u, bins[1:-1]) # assign each pixel to a bin

    bin_unc_mean = []
    bin_err_mean = []
    bin_count = []

    for b in range(num_bins):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        bin_unc_mean.append(u[idx].mean().item()) # map the predicted uncertaintu in the bin, Average model-predicted uncertainty for pixels in bin b.
        bin_err_mean.append(e[idx].mean().item()) # Average actual reconstruction error for the same pixels.
        bin_count.append(idx.sum().item())

    return bin_unc_mean, bin_err_mean, bin_count

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

@torch.no_grad()
def run_inference_and_log(
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

    uncertainty = None
    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        pred_noise, pred_logvar = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(pred_noise, t_tensor, x)
        uncertainty = pred_logvar

    pred_denoised = x
    uncertainty_map = torch.exp(uncertainty)

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
        psnr = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array[mask] - pred_array[mask]) ** 2)
        pear_norm, spear_norm, _ = map_correlations(unc[i][0].numpy(), pred_array, gt_array)
        pear, spear, _ = map_correlations(uncertainty_map[i][0].cpu().detach().numpy(), pred_array, gt_array)
        print(psnr, ssim, mse, pear_norm, spear_norm, pear, spear)
        csv_writer.writerow({'Sample': step * B + i, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim, 'Pearson': pear, 'Spearman': spear,'Pearson_norm': pear_norm, 'Spearman_norm': spear_norm})


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
def run_inference_and_log_v2(
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

    uncertainty = None
    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        pred_noise, pred_logvar = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(pred_noise, t_tensor, x)
        last_t = t_tensor

    pred_denoised = x

    # --- Option A: analytic uncertainty propagation ---
    uncertainty_map = propagate_uncertainty_eps_to_x0(
        pred_logvar_eps=pred_logvar,
        timesteps=last_t,
        scheduler=scheduler,
    )

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

        # Compute metrics
        psnr = compute_psnr(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array, pred_array, data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array - pred_array) ** 2)
        # pear_norm, spear_norm, _ = map_correlations(unc[i][0].numpy(), pred_array, gt_array)
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

        for j in range(5):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] in ["Uncertainty", "Error"] else 'gray')

    plt.tight_layout()
    writer.add_figure("Test/Inference", plt.gcf(), global_step=step)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=False)
    parser.add_argument('--output_dir', type=str,default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", required=False)
    parser.add_argument('--diff_ckpt', type=str, required=False)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--epoch', default=None, type=str)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
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

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
        )

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
            csv_path="test.csv",
            root_dir="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/ND_dataset",
            transform_A=transform,
            transform_B=transform
        )

    elif args.task == "CTPET":
        dataset = Mri2DSlicedataset(args)

    elif args.task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.diff_ckpt = os.path.join(experiment_dir, f"diffusion-ep-{args.epoch}.pth")
    diffusion = networks.init_ddpm_aleatoric(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)

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
    csv_path = os.path.join(experiment_dir, f"{args.experiment_name}_metrics_epoch_{args.epoch}_image_uncertainty_mask.csv")
    csv_path_2 = os.path.join(experiment_dir, f"{args.experiment_name}_metrics_epoch_{args.epoch}_uncertainty_calibration_mask.csv")

    # open both CSV files at the same time and keep them open during inference
    with open(csv_path, mode='w', newline='') as csvfile, \
            open(csv_path_2, mode='w', newline='') as csvfile_2:

        # metrics CSV
        fieldnames = ['Sample', 'MSE', 'PSNR', 'SSIM', 'Pearson_u_norm', 'Spearman_u_norm',  'AUROC_top15_u_norm', 'AUROC_top10_u_norm', 'AUROC_top5_u_norm', 'Pearson_u_unnorm', 'Spearman_u_unnorm',  'AUROC_top15_u_unnorm', 'AUROC_top10_u_unnorm', 'AUROC_top5_u_unnorm']
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

        # calibration CSV
        fieldnames_2 = ['Sample', 'Bin', 'Unc_mean', 'Err_mean', 'Count', 'Type']
        writer_csv_2 = csv.DictWriter(csvfile_2, fieldnames=fieldnames_2)
        writer_csv_2.writeheader()

        for step, batch in enumerate(loader):
            run_inference_and_log_v2(
                diffusion_model=diffusion,
                condition_batch=batch['A'],
                gt_batch=batch['B'],
                writer=writer,
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                csv_writer=writer_csv,
                csv_writer_2=writer_csv_2,
            )

    print(f"✅ Inference complete. Metrics saved to {csv_path}")
