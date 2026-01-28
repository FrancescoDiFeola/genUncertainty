import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from torchvision import transforms
from src.brlp import networks
from inferers import DiffusionInferer
import matplotlib.pyplot as plt
from generative.networks.schedulers import DDIMScheduler
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.MR_to_CT import  MRCTPaired
import numpy as np
# ----------------------------------------------
# ✅ Set environment
# ----------------------------------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


# ----------------------------------------------
# ✅ Loss: Aleatoric (heteroscedastic)
# ----------------------------------------------
def heteroscedastic_loss(pred_mean, pred_logvar, target_noise, min_logvar=-7.0, reg_weight=1e-3):
    # Clamp log variance to prevent overconfidence
    pred_logvar = torch.clamp(pred_logvar, min=min_logvar)

    # Compute precision = 1 / variance
    precision = torch.exp(-pred_logvar)

    # Compute heteroscedastic loss
    base_loss = 0.5 * precision * (target_noise - pred_mean) ** 2 + 0.5 * pred_logvar

    # Optional regularization: penalize too-small variance (i.e., too-large precision)
    reg = precision.mean()  # higher when variance is low
    return base_loss.mean() + reg_weight * reg


# ----------------------------------------------
# ✅ Log to tensorboard
# ----------------------------------------------

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
def sample_and_plot_batch_ddim_aleatoric(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        tag="DDIM_Sampling",
        scheduler=DDIMScheduler,
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
    B, C, H, W = condition_batch.shape
    print(B, C, H, W)

    scheduler.set_timesteps(num_inference_steps)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)
    # model_input = torch.cat([noise, condition_batch], dim=1)  # [B, 2, H, W]
    uncertainty = None

    progress = tqdm(scheduler.timesteps, desc="DDIM Sampling")

    for t in progress:
        t_tensor = torch.tensor([t], device=device).long()

        # Reconstruct input with current latent
        model_input = torch.cat([x, condition_batch], dim=1)

        with autocast(enabled=True):
            dummy_context = torch.zeros((B, 1, 128), device=device)
            pred_noise, pred_logvar = diffusion_model(x=model_input, timesteps=t_tensor, context=dummy_context)
        x, _ = scheduler.step(pred_noise, t_tensor, x)
        uncertainty = pred_logvar

    pred_denoised = x
    uncertainty_map = torch.exp(uncertainty)

    # ---- Plotting ---- #
    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x


    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    # Create figure
    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=5, figsize=(8, 2.5 * num_samples))
    if B == 1:
        axes = [axes]  # make iterable

    for s in range(num_samples):
        images = [ld[s], gt[s], pred[s], unc[s], error[s]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        for j in range(5):
            ax = axes[s][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()

            if titles[j] == "Uncertainty" or titles[j] == "Error":
                ax.imshow(img, cmap='hot')
            else:
                ax.imshow(img, cmap='gray')

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.close()


@torch.no_grad()
def sample_and_plot_batch_ddim_aleatoric_two_pass(
        diffusion_model,
        context_encoder,
        channels,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        tag="DDIM_Sampling",
        scheduler=DDIMScheduler,
        num_inference_steps=50,
):
    """
    DDIM sampling with two-pass inference:
    1. First forward: predict error + uncertainty without context.
    2. Second forward: condition on encoded (error + uncertainty) context using cross-attention.

    Plots [Input | GT | Prediction | Uncertainty | Error].
    """

    diffusion_model.eval()
    B, C, H, W = condition_batch.shape
    scheduler.set_timesteps(num_inference_steps)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    progress = tqdm(scheduler.timesteps, desc="DDIM Two-Pass Sampling")
    uncertainty = None

    for t in progress:
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

        # -----------------------------
        # 🧮 DDIM step update
        # -----------------------------
        x, _ = scheduler.step(pred_error_2, t_tensor, x)
        uncertainty = pred_logvar_2  # store last uncertainty estimate

    # =============================
    # 🎨 Visualization
    # =============================
    pred_denoised = x
    uncertainty_map = torch.exp(uncertainty)

    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x

    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=5, figsize=(8, 2.5 * num_samples))
    if B == 1:
        axes = [axes]

    """
    for i in range(num_samples):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        for j in range(5):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()

            if titles[j] == "Uncertainty" or titles[j] == "Error":
                ax.imshow(img, cmap='hot')
            else:
                ax.imshow(img, cmap='gray')
    """

    for i in range(num_samples):
        images = [ld[i], gt[i], pred[i], unc[i], error[i]]
        titles = ["T1", "T2", "Prediction", "Uncertainty", "Error"]

        for j in range(5):
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

            if titles[j] == "Uncertainty" or titles[j] == "Error":
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
# ----------------------------------------------
# ✅ Training script
# ----------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)
    parser.add_argument('--context_ckpt', default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--backbone', default="UNet", type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--diff_loss_weight', type=float, default=1.0)
    parser.add_argument('--spatial_enc_channels', type=int, default=2)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)

    parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999],help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')

    args = parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Checkpoint directory: {experiment_dir}")

    # -----------------------
    # ✅ Load dataset
    # -----------------------

    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',

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
        )

    elif args.task == "MRtoCT":

        dataset = MRCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_train.csv",
            output_size=256,
        )

    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True)

    # ----------------------------------------------
    # ✅ Load diffusion model
    # ----------------------------------------------
    if args.backbone == "UNet":
        diffusion = networks.init_ddpm_aleatoric_two_forward(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)
        spatial_encoder = networks.init_spatial_context_encoder(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    elif args.backbone == "UViT":
        diffusion = networks.init_uvit_double_output_and_context(img_size=256, in_ch=args.in_ch, out_ch=args.out_ch, checkpoints_path=args.diff_ckpt).to(DEVICE)
        spatial_encoder = networks.init_spatial_context_encoder_UViT(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    print(diffusion)

    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)
        spatial_encoder = torch.nn.DataParallel(spatial_encoder)

    optimizer = torch.optim.AdamW(list(diffusion.parameters()) + list(spatial_encoder.parameters()), lr=args.lr)

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

            noise = torch.randn_like(img_B)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (img_B.size(0),), device=DEVICE).long()

            # === First Forward: Estimate uncertainty map without cross-attention ===
            with torch.no_grad():
                dummy_context = torch.zeros((args.batch_size, 1, 128), device=DEVICE)
                pred_mean_var, _ = inferer(
                    inputs=img_B,
                    concat=img_A,
                    diffusion_model=diffusion,
                    noise=noise,
                    timesteps=timesteps,
                    condition=dummy_context,
                    mode="crossattn",
                )
                uncertainty_map = torch.exp(pred_mean_var[1])  # Convert to variance
                norm_uncertainty_map = norm_percentile(uncertainty_map)  # Normalize for stability

            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)

                # === Encode Uncertainty as Context ===
                if args.spatial_enc_channels == 2:
                    context_input = torch.cat([norm_percentile(pred_mean_var[0]), norm_uncertainty_map], dim=1) # context_input = torch.cat([pred_mean_var[0], pred_mean_var[1]], dim=1) unnormalized
                else:
                    context_input = norm_uncertainty_map  # pred_mean_var[1]

                context_vector = spatial_encoder(context_input)  # shape: (N, 1, cross_attention_dim)
                # Predict noise + log variance
                pred_mean_var, noisy_image = inferer(
                    inputs=img_B,
                    concat=img_A,
                    diffusion_model=diffusion,
                    noise=noise,
                    timesteps=timesteps,
                    condition=context_vector,
                    mode='crossattn'
                )

                # Compute loss
                loss = args.diff_loss_weight * heteroscedastic_loss(pred_mean_var[0], pred_mean_var[1], noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            writer.add_scalar('train/loss', loss.item(), global_counter['train'])
            writer.add_scalar("train/logvar_mean", pred_mean_var[1].mean().item(), global_counter['train'])
            writer.add_scalar("train/logvar_std", pred_mean_var[1].std().item(), global_counter['train'])
            epoch_loss += loss.item()
            global_counter['train'] += 1
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            torch.cuda.empty_cache()

            """
            if step % 2000 == 0:
                sample_and_plot_batch_ddim_aleatoric_two_pass(
                    diffusion_model=diffusion,
                    context_encoder=spatial_encoder,
                    channels=args.spatial_enc_channels,
                    condition_batch=img_A,
                    gt_batch=img_B,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    tag="DDIM_Sampling",
                    scheduler=inference_scheduler,

                )
            """

        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)

        if epoch % 50 == 0:
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(experiment_dir, f'diffusion-ep-{epoch}.pth'))
            torch.save(spatial_encoder.state_dict(), os.path.join(experiment_dir, f'spatial_encoder-ep-{epoch}.pth'))

    print("Training complete.")
