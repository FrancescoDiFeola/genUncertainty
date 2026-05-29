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
from src.VAE.utils.checkpoints_utils import load_checkpoint
from src.brlp.T1_T2_dataset import T1T2Dataset
from src.brlp.CTPET_dataset import CTPETDataset
from src.brlp.CS_dataset import CityscapesColorDataset
from src.brlp.Mri2DSlice_dataset import Mri2DSlicedataset
from src.brlp.ND_dataset import PairedImageDataset
from src.brlp.ldct_hdct_dataset import LDCTHDCTDataset
from src.brlp.MR_to_CT import  MRCTPaired
from src.brlp.CBCTtoCT_dataset import CBCTCTPaired
from src.brlp.motionArtifact_dataset import MotionT1Dataset
# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()

#----------------------------------------------
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


# -----------------------
# ✅ Training script
# -----------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', default="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/", type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)
    parser.add_argument('--context_ckpt', default=None, type=str)
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
    parser.add_argument('--spatial_enc_channels', type=int, default=2)
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

    elif args.task == "T1T2_Oasis":
        dataset = Mri2DSlicedataset(args)
        scaling_factor = 9.404202

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

    spatial_encoder = networks.init_spatial_context_encoder(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    # -----------------------
    # ✅ Load diffusion model
    # -----------------------
    diffusion = networks.init_ddpm_aleatoric_two_forward(args.in_ch, args.out_ch, args.diff_ckpt).to(DEVICE)
    print(diffusion)
    spatial_encoder = networks.init_spatial_context_encoder(channels=args.spatial_enc_channels, cross_attention_dim=128, checkpoints_path=args.context_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)
        autoencoder = torch.nn.DataParallel(autoencoder)
        spatial_encoder = torch.nn.DataParallel(spatial_encoder)

    optimizer = torch.optim.AdamW(list(diffusion.parameters()) + list(spatial_encoder.parameters()), lr=args.lr)

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

            noise = torch.randn_like(img_B_latent)
            timesteps = scheduler.sample_timesteps(img_B_latent)

            img_A_latent = img_A_latent * scaling_factor
            img_B_latent = img_B_latent * scaling_factor

            # === First Forward: Estimate uncertainty map without cross-attention ===
            with torch.no_grad():
                dummy_context = torch.zeros((args.batch_size, 1, 128), device=DEVICE)
                noisy_img_B_latent = scheduler.add_noise(original_samples=img_B_latent, noise=noise, timesteps=timesteps)
                noisy_image_latent = torch.cat([noisy_img_B_latent, img_A_latent], dim=1)
                pred_velocity_mean_var = diffusion(x=noisy_image_latent, timesteps=timesteps, context=dummy_context)
                uncertainty_map = torch.exp(pred_velocity_mean_var[1])  # Convert to variance
                norm_uncertainty_map = norm_percentile(uncertainty_map)  # Normalize for stability

            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)

                # === Encode Uncertainty as Context ===
                if args.spatial_enc_channels == 2:
                    context_input = torch.cat([norm_percentile(pred_velocity_mean_var[0]), norm_uncertainty_map], dim=1)  # context_input = torch.cat([pred_mean_var[0], pred_mean_var[1]], dim=1) unnormalized
                else:
                    context_input = norm_uncertainty_map  # pred_mean_var[1]

                context_vector = spatial_encoder(context_input)  # shape: (N, 1, cross_attention_dim)

                noisy_img_B_latent = scheduler.add_noise(original_samples=img_B_latent, noise=noise, timesteps=timesteps)
                noisy_image_latent = torch.cat([noisy_img_B_latent, img_A_latent], dim=1)

                pred_velocity_mean_var = diffusion(x=noisy_image_latent, timesteps=timesteps, context=context_vector)

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
            """
            if step % 150 == 0:
                sample_and_plot_batch_ddim_aleatoric_two_pass(
                    diffusion_model=diffusion,
                    autoencoder=autoencoder,
                    context_encoder=spatial_encoder,
                    channels=args.spatial_enc_channels,
                    condition_batch=img_A_latent,
                    gt_batch=img_B_latent,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    scheduler=inference_scheduler,
                    scaling=scaling_factor,
                    tag="DDIM_Sampling",

                )
            """

        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)

        if (epoch % 20 == 0 or epoch % 50 == 0):
            # Save the model after each epoch.
            torch.save(diffusion.state_dict(), os.path.join(experiment_dir, f'diffusion-ep-{epoch + args.epoch_start}.pth'))
            torch.save(spatial_encoder.state_dict(), os.path.join(experiment_dir, f'spatial_encoder-ep-{epoch}.pth'))


    print("Training complete.")