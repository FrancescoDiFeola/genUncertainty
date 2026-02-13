import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from torchvision import transforms
import csv
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp import networks
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
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
        z0_s = pred_denoised_latent + eps  # sigma_z0 *
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
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',

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
            # annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
            # annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
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

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=64*64,
        spatial_dim=2
    )

    scheduler.set_timesteps(num_inference_steps=30, device=DEVICE, input_img_size_numel=64*64)

    writer = SummaryWriter(comment=args.experiment_name)
    csv_path = os.path.join(experiment_dir, f"{args.experiment_name}_metrics_ir_epoch_{args.epoch}_K10_ablation_only_small_perturb.csv")

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
                csv_writer=writer_csv,
                K=10,
            )

    print(f"✅ Inference complete. Metrics saved to {csv_path}")
