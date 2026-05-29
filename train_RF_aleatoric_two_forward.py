import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm
from src.brlp import networks
import matplotlib.pyplot as plt
from torchvision import transforms
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.MR_to_CT import  MRCTPaired
from src.brlp.CBCTtoCT_dataset import CBCTCTPaired
from src.brlp.motionArtifact_dataset import MotionT1Dataset

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
def run_inference(
        diffusion_model,
        context_encoder,
        channels,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        tag="DDIM_Sampling",
):
    """
    Sampling with two-pass inference:
    1. First forward: predict error + uncertainty without context.
    2. Second forward: condition on encoded (error + uncertainty) context using cross-attention.

    Plots [Input | GT | Prediction | Uncertainty | Error].
    """

    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

    all_next_timesteps = torch.cat((scheduler.timesteps[1:],torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device)))
    for t, next_t in tqdm(zip(scheduler.timesteps, all_next_timesteps), total = min(len(scheduler.timesteps), len(all_next_timesteps)),):
        t_tensor = torch.tensor([t], device=device).long()
        # -----------------------------
        # 🌀 First Pass: no context (baseline prediction)
        # -----------------------------
        model_input = torch.cat([x, condition_batch], dim=1)

        with autocast(enabled=True):
            dummy_context = torch.zeros((B, 1, 128), device=device)
            predicted_velocity_1, pred_logvar_1 = diffusion_model(x=model_input, timesteps=t_tensor, context=dummy_context)

            # Compute normalized uncertainty map for context
            uncertainty_map = torch.exp(pred_logvar_1)
            norm_uncertainty_map = norm_percentile(uncertainty_map)

            if channels == 2:
                # Concatenate predicted error and normalized uncertainty for conditioning
                context_input = torch.cat([norm_percentile(predicted_velocity_1), norm_uncertainty_map], dim=1)  # (B, 2, H, W). # norm_uncertainty_map
            else:
                context_input = norm_uncertainty_map

            context_vector = context_encoder(context_input)  # (B, 1, 128)

        # -----------------------------
        # 🔁 Second Pass: conditioned refinement
        # -----------------------------
        with autocast(enabled=True):
            predicted_velocity_2, pred_logvar_2 = diffusion_model(x=model_input, timesteps=t_tensor, context=context_vector)

        x, _ = scheduler.step(predicted_velocity_2, t, x, next_t)

        uncertainty = pred_logvar_2

    final_output = x
    uncertainty_map = torch.exp(uncertainty)


    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x

    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = final_output.cpu().detach()
    unc = norm_percentile(uncertainty_map).cpu().detach()
    error = norm_percentile(abs(pred - gt))

    num_samples = B
    fig, axes = plt.subplots(nrows=num_samples, ncols=5, figsize=(8, 2.5 * num_samples))
    if B == 1:
        axes = [axes]

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
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--epoch_start', default=0, type=int)
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

    experiment_dir = os.path.join(f"{args.output_dir}/{args.task}", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Checkpoint directory: {experiment_dir}")

    # args.diff_ckpt = os.path.join(experiment_dir, f"diffusion-ep-{args.epoch_start}.pth")
    # args.context_ckpt = os.path.join(experiment_dir, f"spatial_encoder-ep-{args.epoch_start}.pth")
    # print(args.diff_ckpt, args.context_ckpt)
    # -----------------------
    # ✅ Load dataset
    # -----------------------

    # Load the LDCT/HDCT dataset
    if args.task == "T1T2":
        dataset = T1T2Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',

        )
    elif args.task == "T1motion":

        dataset = MotionT1Dataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_B.csv',
            mode="train",
            motion_range=(0.0, 0.15),
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

    elif args.task == "T1T2_Oasis":

        dataset = Mri2DSlicedataset(args)

    elif args.task == "CBCTtoCT":

        dataset = CBCTCTPaired(
            csv_path= "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/cbct_ct_dataset_train.csv",
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
    diffusion = networks.init_ddpm_aleatoric_two_forward(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)
    print(diffusion)

    spatial_encoder = networks.init_spatial_context_encoder(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)
        spatial_encoder = torch.nn.DataParallel(spatial_encoder)

    optimizer = torch.optim.AdamW(list(diffusion.parameters()) + list(spatial_encoder.parameters()), lr=args.lr)

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=256*256,
        spatial_dim=2
    )

    inference_scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=256*256,
        spatial_dim=2
    )

    inference_scheduler.set_timesteps(num_inference_steps=30, device=DEVICE, input_img_size_numel=256*256)

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
            timesteps = scheduler.sample_timesteps(img_B)

            # === First Forward: Estimate uncertainty map without cross-attention ===
            with torch.no_grad():
                dummy_context = torch.zeros((args.batch_size, 1, 128), device=DEVICE)
                noisy_img_B = scheduler.add_noise(original_samples=img_B, noise=noise, timesteps=timesteps)
                noisy_image = torch.cat([noisy_img_B, img_A], dim=1)
                pred_velocity_mean_var = diffusion(x=noisy_image, timesteps=timesteps, context=dummy_context)
                uncertainty_map = torch.exp(pred_velocity_mean_var[1])  # Convert to variance
                norm_uncertainty_map = norm_percentile(uncertainty_map)  # Normalize for stability


            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)

                # === Encode Uncertainty as Context ===
                if args.spatial_enc_channels == 2:
                    context_input = torch.cat([norm_percentile(pred_velocity_mean_var[0]), norm_uncertainty_map], dim=1) # context_input = torch.cat([pred_mean_var[0], pred_mean_var[1]], dim=1) unnormalized
                else:
                    context_input = norm_uncertainty_map  # pred_mean_var[1]

                context_vector = spatial_encoder(context_input)  # shape: (N, 1, cross_attention_dim)

                noisy_img_B = scheduler.add_noise(original_samples=img_B, noise=noise, timesteps=timesteps)
                noisy_image = torch.cat([noisy_img_B, img_A], dim=1)

                pred_velocity_mean_var = diffusion(x=noisy_image, timesteps=timesteps, context=context_vector)

                # Compute loss
                loss = args.diff_loss_weight * heteroscedastic_loss(pred_velocity_mean_var[0], pred_velocity_mean_var[1], (img_B - noise).float())


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            writer.add_scalar('train/loss', loss.item(), global_counter['train'])
            writer.add_scalar("train/logvar_mean", pred_velocity_mean_var[1].mean().item(), global_counter['train'])
            writer.add_scalar("train/logvar_std", pred_velocity_mean_var[1].std().item(), global_counter['train'])
            epoch_loss += loss.item()
            global_counter['train'] += 1
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            torch.cuda.empty_cache()

            """
            if step % 150 == 0:
                run_inference(
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

        if epoch % 20 == 0:
            # Save the model after each epoch.
            save_epoch = epoch + args.epoch_start
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(experiment_dir, f'diffusion-ep-{save_epoch}.pth'))
            torch.save(spatial_encoder.state_dict(), os.path.join(experiment_dir, f'spatial_encoder-ep-{save_epoch}.pth'))

    print("Training complete.")
