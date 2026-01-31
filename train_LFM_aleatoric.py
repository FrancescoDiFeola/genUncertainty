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
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm
from src.brlp import networks
from inferers import DiffusionInferer
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.VAE.utils.checkpoints_utils import load_checkpoint


# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()

# -----------------------
# ✅ Loss: Aleatoric (heteroscedastic)
# -----------------------
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

def uncertainty_calibration_loss(pred_u_img, err_img, eps=1e-8):
    """
    pred_u_img: (B,1,H,W)  (can be logvar-like or variance-like signal)
    err_img:    (B,1,H,W)  (absolute error map)
    """
    # Make both comparable and stable
    pred = pred_u_img
    pred = pred - pred.mean(dim=(2,3), keepdim=True)
    pred = pred / (pred.std(dim=(2,3), keepdim=True) + eps)

    err = err_img
    err = err - err.mean(dim=(2,3), keepdim=True)
    err = err / (err.std(dim=(2,3), keepdim=True) + eps)

    # MSE on normalized maps
    return F.mse_loss(pred, err)


# -----------------------
# ✅ Training script
# -----------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)
    parser.add_argument('--VAE_ckpt', default=None, type=str)
    parser.add_argument('--unc_decoder_ckpt', default=None, type=str)
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
    parser.add_argument('--uncertainty_loss_weight', type=float, default=0.01)
    parser.add_argument('--in_ch', default=2, type=int)
    parser.add_argument('--out_ch', default=1, type=int)
    parser.add_argument('--uncertainty_calibration', action='store_true', help='enable uncertainty calibration')

    parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999], help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')

    args = parser.parse_args()


    experiment_dir = os.path.join(f"{args.output_dir}/{args.task}", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    args.VAE_ckpt = os.path.join(args.output_dir, args.task, "VAE")
    # -----------------------
    # ✅ Load dataset
    # -----------------------

    scaling_factor = 1
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
        scaling_factor = 7.200933

    elif args.task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation_A='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_LOWDOSE.csv',
            annotation_B='/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/File_annotations/Annotations_D1/Mayo_total_ordinato_FULLDOSE.csv',
        )
        scaling_factor = 7.832608

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

    # *** Load Checkpoints from checkpoint.py ***
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

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=64*64,
        spatial_dim=2
    )

    inference_scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,  # impostato a False nel codice di MAISI
        sample_method='uniform',  # impostato come in MAISI
        use_timestep_transform=True,
        base_img_size_numel=64*64,
        spatial_dim=2
    )

    inference_scheduler.set_timesteps(num_inference_steps=30, device=DEVICE, input_img_size_numel=64*64)

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

            with torch.no_grad():
                _, img_A_latent, _ = autoencoder(img_A)
                _, img_B_latent, _ = autoencoder(img_B)

            img_A_latent = img_A_latent * scaling_factor
            img_B_latent = img_B_latent * scaling_factor

            noise = torch.randn_like(img_B_latent)
            timesteps = scheduler.sample_timesteps(img_B_latent)

            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)
                noisy_img_B_latent = scheduler.add_noise(original_samples=img_B_latent, noise=noise, timesteps=timesteps)
                noisy_image = torch.cat([noisy_img_B_latent, img_A_latent], dim=1)

                pred_velocity_mean_var = diffusion(x=noisy_image, timesteps=timesteps)

                # Compute loss
                loss = args.diff_loss_weight * heteroscedastic_loss(pred_velocity_mean_var[0], pred_velocity_mean_var[1], (img_B_latent - noise).float())


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

            # torch.cuda.empty_cache()

        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)

        if epoch % 50 == 0:
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(experiment_dir, f'diffusion-ep-{epoch + args.epoch_start}.pth'))

    print("Training complete.")