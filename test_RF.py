import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import csv
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.MR_to_CT import MRCTPaired
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
def run_inference(
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

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    all_next_timesteps = torch.cat((scheduler.timesteps[1:],torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for t, next_t in tqdm(zip(scheduler.timesteps, all_next_timesteps), total = min(len(scheduler.timesteps), len(all_next_timesteps)),):
        t_tensor = torch.tensor([t], device=device).long()
        # Reconstruct input with current latent
        model_input = torch.cat([x, condition_batch], dim=1)
        predicted_velocity = diffusion_model(x=model_input, timesteps=t_tensor, context=None)

        x, _ = scheduler.step(predicted_velocity, t, x, next_t)

    final_output = x

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
    pred = final_output.cpu().detach()
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
def run_inference_and_log_MC_sampling(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
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
        for t, next_t in tqdm(zip(scheduler.timesteps, all_next_timesteps), total=min(len(scheduler.timesteps), len(all_next_timesteps)), ):
            t_tensor = torch.tensor([t], device=device).long()
            # Reconstruct input with current latent
            model_input = torch.cat([x, condition_batch], dim=1)
            predicted_velocity = diffusion_model(x=model_input, timesteps=t_tensor, context=None)

            x, _ = scheduler.step(predicted_velocity, t, x, next_t)

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
    parser.add_argument('--MC_sampling', action="store_true")

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
    elif args.task == "MRtoCT":

        dataset = MRCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_test.csv",
            output_size=256,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    diffusion = networks.init_ddpm(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=256*256,
        spatial_dim=2
    )

    scheduler.set_timesteps(num_inference_steps=30, device=DEVICE, input_img_size_numel=256*256)
    writer = SummaryWriter(comment=args.experiment_name)

    if args.MC_sampling:

        csv_path = os.path.join(experiment_dir, f"{args.experiment_name}_metrics_epoch_{args.epoch}_image_uncertainty_MC_sampling.csv")

        # open both CSV files at the same time and keep them open during inference
        with open(csv_path, mode='w', newline='') as csvfile:
            # metrics CSV
            fieldnames = ['Sample', 'MSE', 'PSNR', 'SSIM', 'Pearson_u_norm', 'Spearman_u_norm',  'AUROC_top15_u_norm', 'AUROC_top10_u_norm', 'AUROC_top5_u_norm', 'Pearson_u_unnorm', 'Spearman_u_unnorm',  'AUROC_top15_u_unnorm', 'AUROC_top10_u_unnorm', 'AUROC_top5_u_unnorm']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer_csv.writeheader()

            for step, batch in enumerate(loader):
                run_inference_and_log_MC_sampling(
                    diffusion_model=diffusion,
                    condition_batch=batch['A'],
                    gt_batch=batch['B'],
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    csv_writer=writer_csv,

                )

        print(f"✅ Inference complete. Metrics saved to {csv_path}")
    else:
        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}.csv")

        with open(csv_path, mode='w', newline='') as csvfile:
            fieldnames = ['Sample', 'MSE', 'PSNR', 'SSIM']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer_csv.writeheader()

            for step, batch in enumerate(loader):
                run_inference(
                    diffusion_model=diffusion,
                    condition_batch=batch['A'],
                    gt_batch=batch['B'],
                    writer=writer_csv,
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    csv_writer=writer_csv
                )

        print(f"✅ Inference complete. Metrics saved to {csv_path}")
