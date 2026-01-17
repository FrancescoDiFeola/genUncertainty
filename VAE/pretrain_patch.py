import os
from tqdm import tqdm
from pathlib import Path
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
#from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from monai.networks.nets import PatchDiscriminator
from data.dataset_PATCH import CreateDataloader
from models.autoencoder import Autoencoder
from configs.train_options import TrainOptions
from utils import utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from utils.losses import VAE_Losses
from utils.checkpoints_utils import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#==========================
# FUNCTIONS AND SETTINGS
#==========================


def plot_loss_curves(loss_history, epoch, checkpoint_dir):
    """Salva i grafici delle loss fino all'epoca corrente."""

    checkpoint_dir_path = Path(checkpoint_dir)
    plot_folder = checkpoint_dir_path / "plot"
    loss_folder = plot_folder / "losses"
    loss_folder.mkdir(parents=True, exist_ok=True)

    epochs = loss_history["epochs"]

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Loss Curves - Epoch {epoch}", fontsize=16)

    # Plot 1: Recon Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_history["recon"], label="Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: KL Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss_history["kl"], label="KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KL Divergence")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Perceptual Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, loss_history["perceptual"], label="Perceptual Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Perceptual Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Adversarial Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, loss_history["adv"], label="Adversarial Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Adversarial Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Salva il grafico
    plot_filename = os.path.join(loss_folder, f"loss_curves.png")
    plt.savefig(plot_filename)
    plt.close()
    #print(f"[INFO] Saved loss curves plot to {plot_filename}")


def plot_reconstruction(original, reconstruction, epoch, checkpoint_dir):
    """
    Salva un confronto 2D sui tre piani anatomici (assiale, coronale, sagittale)
    passanti per il centro del volume.
    """

    # Prendi il primo item del batch e rimuovi il canale (C=1)
    with torch.no_grad():
        img_orig = original[0, 0].detach().cpu().float().numpy()
        img_recon = reconstruction[0, 0].detach().cpu().float().numpy()

    # Assumiamo formato [D, H, W] (Depth, Height, Width)
    D, H, W = img_orig.shape

    # Trova le coordinate del centro
    center_D = D // 2
    center_H = H // 2
    center_W = W // 2

    # Estrai le 3 slices per l'originale
    slice_orig_axial = img_orig[center_D, :, :]  # Piano Assiale
    slice_orig_coronal = img_orig[:, center_H, :]  # Piano Coronale
    slice_orig_sagittal = img_orig[:, :, center_W]  # Piano Sagittale

    # Estrai le 3 slices per la ricostruzione
    slice_recon_axial = img_recon[center_D, :, :]
    slice_recon_coronal = img_recon[:, center_H, :]
    slice_recon_sagittal = img_recon[:, :, center_W]

    # Setup per il salvataggio
    checkpoint_dir_path = Path(checkpoint_dir)
    plot_folder = checkpoint_dir_path / "plot"
    recon_folder = plot_folder / "reconstruction"
    recon_folder.mkdir(parents=True, exist_ok=True)

    # Plot: Griglia 3x2
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    plt.suptitle(f"Reconstruction - Epoch {epoch}", fontsize=16)

    # --- Fila 1: Piano Assiale ---
    axes[0, 0].imshow(slice_orig_axial, cmap="gray")
    axes[0, 0].set_title(f"Original - Axial")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(slice_recon_axial, cmap="gray")
    axes[0, 1].set_title(f"Reconstruction - Axial")
    axes[0, 1].axis("off")

    # --- Fila 2: Piano Coronale ---
    axes[1, 0].imshow(slice_orig_coronal, cmap="gray")
    axes[1, 0].set_title(f"Original - Coronal")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(slice_recon_coronal, cmap="gray")
    axes[1, 1].set_title(f"Reconstruction - Coronal")
    axes[1, 1].axis("off")

    # --- Fila 3: Piano Sagittale ---
    axes[2, 0].imshow(slice_orig_sagittal, cmap="gray")
    axes[2, 0].set_title(f"Original - Sagittal")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(slice_recon_sagittal, cmap="gray")
    axes[2, 1].set_title(f"Reconstruction - Sagittal")
    axes[2, 1].axis("off")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Salva il grafico
    plot_filename = os.path.join(recon_folder, f"reconstruction_epoch_{epoch}.png")
    plt.savefig(plot_filename)
    plt.close()
    #print(f"[INFO] Saved reconstruction plot to {plot_filename}")

def save_latest(model, optimizer, epoch, checkpoint_dir, model_name):

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint_data, checkpoint_path)


def expanded_center_mask(mask_tensor):

    B, C, D, H, W = mask_tensor.shape
    cd, ch, cw = D // 2, H // 2, W // 2

    center_ids = mask_tensor[:, 0, cd, ch, cw]
    target_ids = center_ids.view(B, 1, 1, 1, 1)

    binary_mask = (mask_tensor == target_ids) & (target_ids != 0)
    binary_mask = binary_mask.float()

    if voxels > 0:
        padding = voxels
        kernel_size = 1 + 2 * voxels
        binary_mask = F.max_pool3d(
            binary_mask, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )

    return binary_mask


#==========================
# TRAINING LOOP
#==========================


def train_autoencoder(opt):
    # Load Dataset

    print("[INFO] Loading dataset...")
    train_loader = CreateDataloader(opt, shuffle=True)

    if train_loader is None:
        print("[ERROR] Dataset could not be loaded!")
        return

    print("[INFO] Initializing Autoencoder model...")

    num_channels = [32, 64, 128]  # Ensure this is correct
    norm_num_groups = 32


    autoencoder = Autoencoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=[2, 2, 2],
        num_channels=num_channels,
        attention_levels=[False, False, True],
        latent_channels=8,
        norm_num_groups=norm_num_groups,
        norm_eps=1e-6,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        include_fc=False,
        use_combined_linear=False,
        use_flash_attention=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=False,
        print_info=False,
        save_mem=False,
    ).to(device)

    # **Initialize the Discriminator**
    discriminator = PatchDiscriminator(
        spatial_dims=3, num_layers_d=3, channels=32,
        in_channels=1, out_channels=1, norm="INSTANCE"
    ).to(device)


    # Loss handler
    loss_handler = VAE_Losses(
        device, perceptual_weight=opt.perceptual_weight,
        kl_weight=opt.kl_weight, adv_weight=opt.adv_weight
    )

    # **Define Optimizers & Learning Rate Schedulers**
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=opt.lr,
                             eps=1e-6 if opt.amp else 1e-8)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.lr, eps=1e-6 if opt.amp else 1e-8)
    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambda epoch: 0.1 if epoch < 10 else 1.0)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lambda epoch: 0.1 if epoch < 10 else 1.0)

    # **Setup AMP GradScaler**
    scaler_g = GradScaler(enabled=opt.amp)
    scaler_d = GradScaler(enabled=opt.amp)


    checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/src/BASELINE/VAE/checkpoints_prova"

    start_epoch = load_checkpoint(autoencoder, optimizer_g, checkpoint_dir, model_name="autoencoder")
    _ = load_checkpoint(discriminator, optimizer_d, checkpoint_dir, model_name="discriminator")
    print(f"Resuming training from epoch {start_epoch}")

    loss_history = {
        "epochs": [],
        "recon": [],
        "kl": [],
        "perceptual": [],
        "adv": []
    }

    print("[INFO] Starting training...")

    for epoch in range(start_epoch, opt.n_epochs):
        autoencoder.train()
        discriminator.train()
        total_loss = {"recon": 0, "kl": 0, "perceptual": 0, "adv": 0}

        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", leave=True)

        for i, batch in enumerate(epoch_iter):

            image = batch['image'].to(device)
            mask = batch['mask'].to(device)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            with autocast(enabled=True, dtype=torch.bfloat16):

                z, mu, log_var, reconstruction = autoencoder(image)

                logits_reconstruction = discriminator(reconstruction.detach())[-1]  # Detach to avoid generator gradients
                logits_real = discriminator(image.detach())[-1]

                loss_d = (
                        loss_handler.adv_loss(logits_reconstruction, target_is_real=False, for_discriminator=True) +
                        loss_handler.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                )

                scaler_d.scale(loss_d).backward()
                # **Gradient Accumulation for Discriminator**
                if (i + 1) % opt.gradient_accumulation_steps == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()  # Reset gradients

                # **Compute Generator Losses**
                losses, loss_g = loss_handler.compute_losses(
                    reconstruction, image, mu, log_var, discriminator, mask
                )

                scaler_g.scale(loss_g).backward()

                # **Gradient Accumulation for Generator**
                if (i + 1) % opt.gradient_accumulation_steps == 0:
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad()  # Reset gradients


            # Track total loss for per-epoch logging
            for key in total_loss:
                if key in losses:
                    total_loss[key] += losses[key].item()
                if key == "adv":
                    total_loss["adv"] += loss_d.item()


        # writer.flush()
        # Step the learning rate scheduler after each epoch
        scheduler_g.step()
        scheduler_d.step()

        avg_loss = {k: v / len(train_loader) for k, v in total_loss.items()}

        loss_history["epochs"].append(epoch + 1)
        loss_history["recon"].append(avg_loss['recon'])
        loss_history["kl"].append(avg_loss['kl'])
        loss_history["perceptual"].append(avg_loss['perceptual'])
        loss_history["adv"].append(avg_loss['adv'])



        plot_loss_curves(loss_history, epoch + 1, checkpoint_dir)

        save_latest(autoencoder, optimizer_g, epoch, checkpoint_dir, model_name="autoencoder")
        save_latest(discriminator, optimizer_d, epoch, checkpoint_dir, model_name="discriminator")

        if (epoch + 1) % 50 == 0:
            if not torch.isnan(reconstruction).any():
                plot_reconstruction(image, reconstruction, epoch + 1, checkpoint_dir)
            else:
                print(f"[WARNING] Skipping reconstruction plot at epoch {epoch + 1} due to NaNs in reconstruction.")

            save_checkpoint(autoencoder, optimizer_g, epoch, checkpoint_dir, model_name="autoencoder")
            save_checkpoint(discriminator, optimizer_d, epoch, checkpoint_dir, model_name="discriminator")


        print(
            f"Epoch [{epoch + 1}/{opt.n_epochs}], Recon : {avg_loss['recon']:.6f}, KL Loss: {avg_loss['kl']:.6f}, Perceptual Loss: {avg_loss['perceptual']:.6f}, Adv Loss: {avg_loss['adv']:.6f}")




    print("[INFO] Training Complete! Model saved to pretrained/checkpoints")


if __name__ == "__main__":
    opt = TrainOptions()
    train_autoencoder(opt),