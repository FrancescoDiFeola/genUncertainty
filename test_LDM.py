import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from torchvision import transforms
from generative.networks.schedulers import DDIMScheduler
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp import networks
from src.VAE.utils.checkpoints_utils import load_checkpoint
from src.inference.inference_LDM import *
from src.inference.utils import initialize_writers
from src.brlp.MR_to_CT import MRCTPaired
from src.brlp.CBCTtoCT_dataset import CBCTCTPaired
from src.brlp.motionArtifact_dataset import MotionT1Dataset

# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


@torch.no_grad()
def run_inference_and_log(
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
def run_inference_and_log_MC_sampling(
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=False)
    parser.add_argument('--output_dir', type=str, default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", required=False)
    parser.add_argument('--diff_ckpt', type=str, required=False)
    parser.add_argument('--VAE_ckpt', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=str)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--motion_level', default=1, type=str)
    parser.add_argument('--analysis', type=str, required=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)
    parser.add_argument('--MC_sampling', action="store_true")
    parser.add_argument('--n_sampling', default=4, type=int)

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
    args.VAE_ckpt = os.path.join(args.output_dir, args.task, "VAE")

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    scaling_factor = 1
    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        # dataset = Mri2DSlicedataset(args)
        # scaling_factor = 9.404202

        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
        )
        scaling_factor = 9.404202

    elif args.task == "T1motion":

        dataset = MotionT1Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A_test.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B_test.csv',
            mode="test",
            fixed_motion_level = float(args.motion_level),
        )  # test_dataset_lvl_0 = T1T2Dataset(..., mode="test", fixed_motion_level=0.0
        scaling_factor = 5.634654

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
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_lowdose_GAN_D2_nuovo_ordinato.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D2/annotations_test_fulldose_GAN_D2_nuovo_ordinato.csv',
        )
        scaling_factor = 7.832608

    elif args.task == "T1T2_Oasis":
        dataset = Mri2DSlicedataset(args)
        scaling_factor = 9.404202

    elif args.task == "MRtoCT":

        dataset = MRCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_test.csv",
            output_size=256,
        )
        scaling_factor = 6.640712

    elif args.task == "CBCTtoCT":

        dataset = CBCTCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/cbct_ct_dataset_test.csv",
            output_size=256,
        )
        scaling_factor=9.744896

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

    diffusion = networks.init_ddpm(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)
        autoencoder = torch.nn.DataParallel(autoencoder)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    writer = SummaryWriter(comment=args.experiment_name)


    if args.analysis == "sparsification":

        csv_path = os.path.join(experiment_dir, f"sparsification_S_{args.n_sampling}_epoch_{args.epoch}_motion_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "metrics":

        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_image_uncertainty_train.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)

    elif args.analysis == "metrics_no_uncertainty":

        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}__motion_{args.motion_level}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    elif args.analysis == "uncertainty_eval":
        csv_path = os.path.join(experiment_dir, f"metrics_epoch_{args.epoch}_uncertainty_eval_{args.motion_level}_S_{args.n_sampling}.csv")
        writer_csv = initialize_writers(csv_path, writer_type=args.analysis)[1]

    if args.MC_sampling:

        for step, batch in tqdm(enumerate(loader)):
            img_A = batch["A"].to(DEVICE)
            img_B = batch["B"].to(DEVICE)

            with torch.no_grad():
                _, img_A_latent, _ = autoencoder(img_A)

            img_A_latent = img_A_latent * scaling_factor

            if args.analysis == "sparsification":

                run_inference_LDM_vanilla_and_log_MC_sampling_sparsification(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    condition_batch=img_A_latent,
                    gt_batch=img_B,
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    scaling=scaling_factor,
                    csv_writer=writer_csv,
                    n_sampling=args.n_sampling,
                )

            elif args.analysis == "metrics":
                run_inference_LDM_vanilla_and_log_MC_sampling(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    condition_batch=img_A_latent,
                    gt_batch=img_B,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    scaling=scaling_factor,
                    csv_writer=writer_csv
                )
            elif args.analysis == "uncertainty_eval":
                run_inference_LDM_vanilla_and_log_MC_sampling_uncertainty_eval(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    condition_batch=img_A_latent,
                    gt_batch=img_B,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    scheduler=scheduler,
                    scaling=scaling_factor,
                    csv_writer=writer_csv,
                    n_sampling=args.n_sampling,
                )
        print(f"✅ Inference complete. Metrics saved to {csv_path}")

    else:

        for step, batch in tqdm(enumerate(loader)):
            img_A = batch["A"].to(DEVICE)
            img_B = batch["B"].to(DEVICE)

            with torch.no_grad():
                _, img_A_latent, _ = autoencoder(img_A)

            img_A_latent = img_A_latent * scaling_factor

            run_inference_LDM_vanilla_and_log(
                diffusion_model=diffusion,
                autoencoder=autoencoder,
                condition_batch=img_A_latent,
                gt_batch=img_B,
                writer=writer,
                step=step,
                device=DEVICE,
                scheduler=scheduler,
                scaling=scaling_factor,
                csv_writer=writer_csv
            )

        print(f"✅ Inference complete. Metrics saved to {csv_path}")

