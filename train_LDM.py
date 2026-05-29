import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
# from torch.utils.data import DataLoader
from monai.data import DataLoader
from torchvision import transforms
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from src.brlp import networks
from inferers import DiffusionInferer
import numpy as np
import matplotlib.pyplot as plt
from generative.networks.schedulers import DDIMScheduler
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.MR_to_CT import  MRCTPaired
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.VAE.utils.checkpoints_utils import load_checkpoint
from src.brlp.CBCTtoCT_dataset import CBCTCTPaired
from src.brlp.motionArtifact_dataset import MotionT1Dataset

# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


# -----------------------
# ✅ Log to tensorboard
# -----------------------


@torch.no_grad()
def sample_and_plot_batch_ddim(
        diffusion_model,
        autoencoder,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        scaling,
        tag="DDIM_Sampling",
        num_training_steps=1000,
        num_inference_steps=50,
        beta_start=0.0015,
        beta_end=0.0205,
):
    """
    DDIM sampling + tensorboard batch display with uncertainty.
    Plots [LD | GT | Prediction | Uncertainty] per row.
    """

    diffusion_model.eval()
    B, _, _, _ = condition_batch.shape

    scheduler.set_timesteps(num_inference_steps)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    progress = tqdm(scheduler.timesteps, desc="DDIM Sampling")

    for t in progress:
        t_tensor = torch.tensor([t], device=device).long()

        # timestep tensor must match batch size: [B]
        # t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # Reconstruct input with current latent
        model_input = torch.cat([x, condition_batch], dim=1)

        with autocast(enabled=True):
            pred_noise = diffusion_model(x=model_input, timesteps=t_tensor, context=None)

        x, _ = scheduler.step(pred_noise, t_tensor, x)

    pred_denoised_latent = x

    pred_denoised = autoencoder.decode(pred_denoised_latent/scaling)
    gt_batch = autoencoder.decode(gt_batch/scaling)
    condition_batch = autoencoder.decode(condition_batch/scaling)

    # ---- Plotting ---- #
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
    error = norm_percentile(abs(pred - gt))

    # Create figure
    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(8, 2.5 * num_samples))
    if B == 1:
        axes = [axes]  # make iterable
    """
    for i in range(num_samples):
        images = [ld[i], gt[i], pred[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Error"]

        for j in range(4):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()

            if titles[j] == "Error":
                ax.imshow(img, cmap='hot')
            else:
                ax.imshow(img, cmap='gray')
    """

    for i in range(num_samples):
        images = [ld[i], gt[i], pred[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Error"]

        for j in range(4):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])

            img = images[j].cpu().numpy()  # shape: (C, H, W) or (1, H, W)
            if img.ndim == 3:
                # (C, H, W) -> (H, W) or (H, W, 3)
                if img.shape[0] == 1:
                    img = img[0]  # (H, W) grayscale
                elif img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))  # (H, W, 3) RGB

            if titles[j] == "Error":
                # always show error as grayscale
                if img.ndim == 3 and img.shape[2] == 3:
                    err_gray = np.mean(img, axis=2)
                    ax.imshow(err_gray, cmap="hot")
                else:
                    ax.imshow(img, cmap="hot")
            else:
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)  # RGB

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.close()


# -----------------------
# ✅ Training script
# -----------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)
    parser.add_argument('--VAE_ckpt', default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=305, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--epoch_start', default=0, type=float)
    parser.add_argument('--diff_loss_weight', type=float, default=1.0)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)

    parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999], help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')

    args = parser.parse_args()

    experiment_dir = os.path.join(f"{args.output_dir}/{args.task}", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    args.VAE_ckpt = os.path.join(args.output_dir, args.task, "VAE")

    # ----------------------- #
    # ✅ Load dataset
    # ----------------------- #
    scaling_factor = 1
    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',

        )
        scaling_factor = 9.404202

    elif args.task == "T1motion":

        dataset = MotionT1Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',
            mode="train",
            motion_range=(0.0, 0.15),
        )  # test_dataset_lvl_0 = T1T2Dataset(..., mode="test", fixed_motion_level=0.0)
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
        scaling_factor = 7.200933

    elif args.task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
        )
        scaling_factor = 7.832608

    elif args.task == "T1T2_Oasis":
        dataset = Mri2DSlicedataset(args)
        scaling_factor = 9.404202

    elif args.task == "MRtoCT":

        dataset = MRCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_train.csv",
            output_size=256,
        )
        scaling_factor = 6.640712

    elif args.task == "CBCTtoCT":

        dataset = CBCTCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/cbct_ct_dataset_train.csv",
            output_size=256,
        )
        scaling_factor=9.744896

    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True)

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

    # -----------------------
    # ✅ Load diffusion model
    # -----------------------
    diffusion = networks.init_ddpm(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)
        autoencoder = torch.nn.DataParallel(autoencoder)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    inference_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        # At inference time, even if you’re doing DDIM with fewer steps (e.g. 50), you must pass the same num_train_timesteps to the inference scheduler as you used for training, because it defines the same discrete time grid the model was trained on.
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    scaler = GradScaler()
    writer = SummaryWriter(comment=args.experiment_name)

    global_counter = {'train': 0}

    # -----------------------
    # ✅ Training loop
    # -----------------------
    for epoch in range(args.n_epochs):
        diffusion.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            img_A = batch["A"].to(DEVICE)  # Low-dose CT
            img_B = batch["B"].to(DEVICE)  # High-dose CT
            print(img_A.shape, img_B.shape)
            with torch.no_grad():
                _, img_A_latent, _ = autoencoder(img_A)
                _, img_B_latent, _ = autoencoder(img_B)

            img_A_latent = img_A_latent * scaling_factor
            img_B_latent = img_B_latent * scaling_factor

            noise = torch.randn_like(img_B_latent)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (img_B_latent.size(0),), device=DEVICE).long()

            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)

                # Predict noise + log variance
                noise_pred, _ = inferer(
                    inputs=img_B_latent,
                    concat=img_A_latent,
                    diffusion_model=diffusion,
                    noise=noise,
                    timesteps=timesteps,
                    condition=img_A_latent,
                    mode='concat'
                )
                # Compute loss
                loss = F.mse_loss(noise.float(), noise_pred.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            writer.add_scalar('train/loss', loss.item(), global_counter['train'])
            epoch_loss += loss.item()
            global_counter['train'] += 1
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            # torch.cuda.empty_cache()

            if step % 150 == 0:
                sample_and_plot_batch_ddim(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    condition_batch=img_A_latent,
                    gt_batch=img_B_latent,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    scheduler=inference_scheduler,
                    scaling=scaling_factor,
                    tag="DDIM_Sampling",

                )


        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)

        if epoch % 20 == 0:
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(experiment_dir, f'diffusion-ep-{epoch}.pth'))

    print("Training complete.")
