import os
from tqdm import tqdm
from pathlib import Path
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from monai.networks.nets import PatchDiscriminator
from src.VAE.data.dataset_Denoising import LDCTHDCTDataset
# from models.autoencoder import Autoencoder
from monai.networks.nets.autoencoderkl import AutoencoderKL
from src.VAE.configs.train_options import TrainOptions
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from src.VAE.utils.losses import VAE_Losses
from monai.data import DataLoader, CacheDataset, DataLoader
from src.VAE.utils.checkpoints_utils import save_checkpoint, load_checkpoint
from src.VAE.data.dataset_MRtoCT import MRCTSingleImageDataset
from src.VAE.data.dataset_T1T2 import T1T2Dataset
from src.VAE.data.dataset_CTPET import CTPETDataset
from src.VAE.data.dataset_CBCTtoCT import CBCTCTSingleImageDataset
from src.VAE.data.dataset_motionT1 import MotionT1VaeDataset
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_latent_space(z_bl, z_fu, epoch, checkpoint_dir):

    # Convert latent tensors to numpy
    z_bl_np = z_bl.detach().cpu().numpy().reshape(len(z_bl), -1)
    z_fu_np = z_fu.detach().cpu().numpy().reshape(len(z_fu), -1)

    # Combine for joint t-SNE embedding
    z_combined = np.concatenate([z_bl_np, z_fu_np])

    if np.isnan(z_combined).any():
        print(f"[WARNING] NaNs found in latent space at epoch {epoch} — skipping t-SNE plot.")
        return

    # Use safe perplexity for t-SNE
    perplexity_value = min(30, len(z_combined) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    z_embedded = tsne.fit_transform(z_combined)

    # Save plot in district-named folder
    checkpoint_dir_path = Path(checkpoint_dir)
    plot_folder = checkpoint_dir_path / "plot"
    latent_space_folder = plot_folder / "latent_space"
    latent_space_folder.mkdir(parents=True, exist_ok=True)

    # Plot the latent points
    plt.figure(figsize=(6, 5))
    plt.scatter(z_embedded[:len(z_bl), 0], z_embedded[:len(z_bl), 1], label="BASELINE", alpha=0.6)
    plt.scatter(z_embedded[len(z_bl):, 0], z_embedded[len(z_bl):, 1], label="FOLLOWUP", alpha=0.6)

    # Draw dashed lines between corresponding CT–PET latent vectors
    for idx in range(len(z_bl)):
        bl_point = z_embedded[idx]
        fu_point = z_embedded[len(z_bl) + idx]
        plt.plot([bl_point[0], fu_point[0]], [bl_point[1], fu_point[1]], 'w--', alpha=0.3, linewidth=0.8)

    plt.legend()
    plt.title(f"Latent Space - Epoch {epoch}")
    plt.tight_layout()

    # Save and log the image
    plot_filename = os.path.join(latent_space_folder, f"latent_space_epoch_{epoch}.png")
    plt.savefig(plot_filename)

    #if writer:
    #    img = plt.imread(plot_filename)
    #   writer.add_image("Latent_Space", torch.tensor(img).permute(2, 0, 1), epoch)

    plt.close()


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
    plt.plot(epochs, loss_history["perceptual"], label="Perceptual Loss")  # loss_history["adv"]
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Perceptual Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Salva il grafico
    plot_filename = os.path.join(loss_folder, f"loss_curves_epoch_{epoch}.png")
    plt.savefig(plot_filename)
    plt.close()
    #print(f"[INFO] Saved loss curves plot to {plot_filename}")

"""  # 3D function
def plot_reconstruction(original, reconstruction, epoch, checkpoint_dir):
    
    # Salva un confronto 2D sui tre piani anatomici (assiale, coronale, sagittale)
    # passanti per il centro del volume.
    

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
"""

def plot_reconstruction(original, reconstruction, epoch, checkpoint_dir):
    """
    Salva un confronto 2D tra immagine originale e ricostruzione.
    Assumiamo input con shape [B, C, H, W].
    """

    import os
    from pathlib import Path
    import torch
    import matplotlib.pyplot as plt

    # Prendi il primo item del batch e rimuovi il canale (C=1)
    with torch.no_grad():
        img_orig = original[0, 0].detach().cpu().float().numpy()
        img_recon = reconstruction[0, 0].detach().cpu().float().numpy()

    # Setup per il salvataggio
    checkpoint_dir_path = Path(checkpoint_dir)
    plot_folder = checkpoint_dir_path / "plot"
    recon_folder = plot_folder / "reconstruction"
    recon_folder.mkdir(parents=True, exist_ok=True)

    # Plot: griglia 1x2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f"Reconstruction - Epoch {epoch}", fontsize=16)

    # Originale
    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Ricostruzione
    axes[1].imshow(img_recon, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Salva il grafico
    plot_filename = os.path.join(
        recon_folder, f"reconstruction_epoch_{epoch}.png"
    )
    plt.savefig(plot_filename)
    plt.close()

def train_autoencoder(opt):
    # Load Dataset

    print("[INFO] Loading dataset...")

    if opt.task == "denoising":
        dataset = LDCTHDCTDataset(
            annotation='/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/src/VAE/csvs/Mayo_total_stacked_shuffled.csv',
        )

    elif opt.task == "MRtoCT":
        dataset = MRCTSingleImageDataset(csv_path="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/mr_ct_dataset_train.csv")

    elif opt.task == "T1T2":
        dataset = T1T2Dataset("/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/src/VAE/csvs/T1T2_train.csv")

    elif opt.task == "CTPET":
        dataset = CTPETDataset(opt)

    elif opt.task == "T1T2_Oasis":
        dataset = CTPETDataset(opt)

    elif opt.task == "T1motion":

        dataset = MotionT1VaeDataset(
            annotation_T1= '/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/Data/annotations_A.csv',
            mode="train",
            motion_range=(0.0, 0.15),  # moderate corruption
            base_seed=1234,
            include_clean=True  # mix clean + corrupted
        )

    elif opt.task == "CBCTtoCT":
        dataset = CBCTCTSingleImageDataset(csv_path="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/cbct_ct_dataset_train.csv")

    train_loader = DataLoader(dataset=dataset,
                              batch_size=opt.batchSize,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)

    if train_loader is None:
        print("[ERROR] Dataset could not be loaded!")
        return

    print("[INFO] Initializing Autoencoder model...")

    num_channels = [32, 64, 128]  # Ensure this is correct
    norm_num_groups = 32

    """
    autoencoder = Autoencoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=[2, 2, 2],
        num_channels=num_channels,
        attention_levels=[False, False, False],
        latent_channels=3,
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
    """
    device = torch.device("cuda")

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
    autoencoder = autoencoder.to(device)


    # **Initialize the Discriminator**
    discriminator = PatchDiscriminator(
        spatial_dims=2, num_layers_d=3, channels=64, in_channels=1, out_channels=1, norm="INSTANCE"
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        autoencoder = nn.DataParallel(autoencoder)
        discriminator = nn.DataParallel(discriminator)

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

    # **Initialize Tensorboard Logging**
    #writer = SummaryWriter(comment="eventlog_for_vae_training")
    #avgloss = utils.AverageLoss()
    #total_counter = 0  # print(f"TensorBoard logging directory: {writer.log_dir}")
    #checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/src/VAE/checkpoints"
    checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/T1motion/VAE"
    # checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/checkpoints/CBCTtoCT/VAE"

    # **Load Checkpoints from checkpoint.py**
    start_epoch = load_checkpoint(autoencoder, optimizer_g, checkpoint_dir, model_name="autoencoder")  # opt
    _ = load_checkpoint(discriminator, optimizer_d, checkpoint_dir, model_name="discriminator")
    print(f"Resuming training from epoch {start_epoch}")

    loss_history = {
        "epochs": [],
        "recon": [],
        "kl": [],
        "perceptual": [],
        "adv": []
    }

    # apply_gradient_checkpointing(autoencoder.encoder)
    # apply_gradient_checkpointing(autoencoder.decoder)
    print("[INFO] Starting training...")

    for epoch in range(start_epoch, opt.n_epochs):
        autoencoder.train()
        discriminator.train()
        total_loss = {"recon": 0, "kl": 0, "perceptual": 0, "adv": 0}

        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", leave=True)

        for i, batch in enumerate(epoch_iter):
            # Free unused memory before processing a new batch
            #torch.cuda.empty_cache()
            #gc.collect()

            image = batch['img'].to(device)

            # img_ct = torch.empty(1, 1, 200, 200, 200).to(device)
            # img_pet = torch.empty(1, 1, 150, 150, 150).to(device)
            #print(f"shape of baseline image:{img_bl.shape}, shape of followup image: {img_fu.shape}")

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            with autocast(enabled=True, dtype=torch.bfloat16):
                # **Encode CT & PET images into shared latent space**
                reconstruction, mu, log_var = autoencoder(image)  #  z, mu, log_var, reconstruction

                #print(f"Min valore input: {img_fu.min().item():.4f}, Max valore input: {img_fu.max().item():.4f}")
                #print(f"Min valore output: {recon_fu.min().item():.4f}, Max valore output: {recon_fu.max().item():.4f}")

                logits_reconstruction = discriminator(reconstruction.detach())[-1]  # Detach to avoid generator gradients
                logits_real = discriminator(image.detach())[-1]

                loss_d = (
                        loss_handler.adv_loss(logits_reconstruction, target_is_real=False, for_discriminator=True) +
                        loss_handler.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                )
                #print(f"loss_d_fu:{loss_d_fu}")

                # **Backpropagation for Discriminator**

                scaler_d.scale(loss_d).backward()
                # **Gradient Accumulation for Discriminator**
                if (i + 1) % opt.gradient_accumulation_steps == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()  # Reset gradients


                # **Compute Generator Losses**
                losses, loss_g = loss_handler.compute_losses(
                    reconstruction, image, mu, log_var, discriminator
                )

                #print("DEBUG: Losses computed successfully!")  # Debugging line
                #print(f"losses: {losses}, loss_g: {loss_g}")

                #print(f"Recon FU: {recon_fu.shape}")
                #print(f"Latent Mean: {mu_bl.shape}, Latent Mean: {mu_fu.shape}, Log Variance: {log_var_bl.shape}, Log Variance: {log_var_fu.shape}")

                # **Backpropagation for Generator**
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



        if (epoch) % 50 == 0:
        #    if not (torch.isnan(z_bl).any() or torch.isnan(z_fu).any()):
        #        plot_latent_space(z_bl, z_fu, epoch + 1, checkpoint_dir=checkpoint_dir)
        #    else:
        #        print(f"[WARNING] Skipping latent space plot at epoch {epoch + 1} due to NaNs in z_bl or z_fu.")

            plot_loss_curves(loss_history, epoch + 1, checkpoint_dir)

        if (epoch) % 20 == 0:
            if not torch.isnan(reconstruction).any():
                plot_reconstruction(image, reconstruction, epoch + 1, checkpoint_dir)
            else:
                print(f"[WARNING] Skipping reconstruction plot at epoch {epoch + 1} due to NaNs in reconstruction.")

        print(
            f"Epoch [{epoch + 1}/{opt.n_epochs}], Recon : {avg_loss['recon']:.6f}, KL Loss: {avg_loss['kl']:.6f}, Perceptual Loss: {avg_loss['perceptual']:.6f}") # , Adv Loss: {avg_loss['adv']:.6f}")

        # **Save Model Checkpoint**
        if (epoch) % 50 == 0:
            save_checkpoint(autoencoder, optimizer_g, epoch, checkpoint_dir, model_name="autoencoder")
            save_checkpoint(discriminator, optimizer_d, epoch, checkpoint_dir, model_name="discriminator")

    print("[INFO] Training Complete! Model saved to pretrained/checkpoints")


if __name__ == "__main__":
    opt = TrainOptions()
    train_autoencoder(opt),