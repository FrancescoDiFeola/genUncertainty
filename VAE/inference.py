import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
from pathlib import Path

from torch.cuda.amp import autocast
from monai.inferers import sliding_window_inference
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from configs.test_options import TestOptions
from models.autoencoder import Autoencoder
from data.dataset_BASE import CreateDataloader
from utils.checkpoints_utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def compute_metrics(image, reconstruction):
    image_np = image.squeeze().cpu().float().numpy()
    reconstruction_np = reconstruction.squeeze().cpu().float().numpy()

    data_range = 1.0

    SSIM = ssim(image_np, reconstruction_np, data_range=data_range)
    PSNR = psnr(image_np, reconstruction_np, data_range=data_range)
    MAE = np.mean(np.abs(image_np - reconstruction_np))

    return SSIM, PSNR, MAE


def save_and_visualize_results(image, reconstruction, output_dir, filename, index, batch=None, save_nifti=False, save_png=True, visualize_every=6):

    output_dir_path = Path(output_dir)
    nifti_folder = output_dir_path / "nifti"
    comparison_folder = output_dir_path / "comparison"
    nifti_folder.mkdir(parents=True, exist_ok=True)
    comparison_folder.mkdir(parents=True, exist_ok=True)

    """
    if save_nifti:
        recon_np = reconstruction.squeeze().cpu().float().numpy()

        if batch is not None and "affine" in batch:
            affine = batch["affine"].cpu().numpy()[0]
        else:
            affine = np.eye(4)

        recon_img = nib.Nifti1Image(recon_np, affine)
        nib.save(recon_img, os.path.join(nifti_folder, f"recon_{index:03d}.nii.gz"))
    """
    if save_nifti:
        recon_np = reconstruction.squeeze().cpu().float().numpy()

        # Tentativo robusto di recuperare l'affine matrix
        affine = np.eye(4)
        if batch is not None:
            if "A_meta_dict" in batch and "affine" in batch["A_meta_dict"]:
                # MONAI standard move to meta_dict
                affine = batch["A_meta_dict"]["affine"][0].cpu().numpy()
            elif "affine" in batch:
                # Affine diretto
                affine = batch["affine"][0].cpu().numpy()

        recon_img = nib.Nifti1Image(recon_np, affine)

        # USIAMO IL FILENAME REALE QUI
        nib.save(recon_img, os.path.join(nifti_folder, f"{filename}.nii.gz"))


    if save_png and (index % visualize_every == 0):

        img_gt_np = image[0, 0].detach().cpu().float().numpy()
        recon_np = reconstruction[0, 0].detach().cpu().float().numpy()

        D, H, W = img_gt_np.shape


        center_D = D // 2
        center_H = H // 2
        center_W = W // 2


        slice_orig_axial = img_gt_np[center_D, :, :]  # Piano Assiale
        slice_orig_coronal = img_gt_np[:, center_H, :]  # Piano Coronale
        slice_orig_sagittal = img_gt_np[:, :, center_W]  # Piano Sagittale


        slice_recon_axial = recon_np[center_D, :, :]
        slice_recon_coronal = recon_np[:, center_H, :]
        slice_recon_sagittal = recon_np[:, :, center_W]


        fig, axes = plt.subplots(3, 2, figsize=(10, 13))
        plt.suptitle(f"Reconstruction - Sample {index}", fontsize=16)

        # --- Fila 1: Piano Assiale ---
        axes[0, 0].imshow(slice_orig_axial, cmap="gray")
        axes[0, 0].set_title("Original - Axial")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(slice_recon_axial, cmap="gray")
        axes[0, 1].set_title("Recon - Axial")
        axes[0, 1].axis("off")

        # --- Fila 2: Piano Coronale ---
        axes[1, 0].imshow(slice_orig_coronal, cmap="gray")
        axes[1, 0].set_title("Original - Coronal")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(slice_recon_coronal, cmap="gray")
        axes[1, 1].set_title("Recon - Coronal")
        axes[1, 1].axis("off")

        # --- Fila 3: Piano Sagittale ---
        axes[2, 0].imshow(slice_orig_sagittal, cmap="gray")
        axes[2, 0].set_title("Original - Sagittal")
        axes[2, 0].axis("off")

        axes[2, 1].imshow(slice_recon_sagittal, cmap="gray")
        axes[2, 1].set_title("Recon - Sagittal")
        axes[2, 1].axis("off")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))


        plot_filename = os.path.join(comparison_folder, f"comparison_{index:03d}.png")
        plt.savefig(plot_filename)
        plt.close()


class AutoencoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z, mu, logvar, reconstruction = self.model(x)
        return reconstruction


def test_autoencoder(opt):

    output_path = "/home/lcarusone/TesiMagistrale/src/VAE"
    output_dir = os.path.join(output_path, "inference_base")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading dataset...")
    test_loader = CreateDataloader(opt, shuffle=False)

    if test_loader is None:
        print("[ERROR] Dataset could not be loaded!")
        return

    print("[INFO] Initializing Autoencoder model...")


    num_channels = [32, 64, 128]
    norm_num_groups = 32


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


    checkpoint_dir = "/home/lcarusone/TesiMagistrale/src/VAE/checkpoints_base"
    _ = load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=checkpoint_dir, opt=opt, model_name="autoencoder")
    print("[INFO] Pesi caricati con successo.")

    autoencoder = AutoencoderWrapper(autoencoder).to(device)
    autoencoder.eval()


    metrics_list = []


    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inferenza")):
            image = batch['A'].to(device)

            if 'a_name' in batch:
                filename = batch['a_name'][0]
            else:
                filename = f"patient_{i:03d}"

            with autocast(enabled=True, dtype=torch.bfloat16):
                #outputs = autoencoder(image)
                #reconstruction = outputs[-1]
                reconstruction = sliding_window_inference(
                    image,
                    roi_size=(64, 64, 64),
                    sw_batch_size=16,
                    predictor=autoencoder,
                    overlap=0.625,
                    mode="gaussian",
                    sigma_scale=0.125
                )



            #print(f"Min valore input: {image.min().item():.4f}, Max valore input: {image.max().item():.4f}")
            #print(f"Min valore output: {reconstruction.min().item():.4f}, Max valore output: {reconstruction.max().item():.4f}")
            #print(f"Media input: {img_fu.mean().item():.4f}, Std input: {img_fu.std().item():.4f}")
            #print(f"Media output: {recon_fu.mean().item():.4f}, Std output: {recon_fu.std().item():.4f}")



            #SSIM, PSNR, MAE = compute_metrics(img_fu, recon_fu)
            SSIM, PSNR, MAE = compute_metrics(image, reconstruction)
            metrics_list.append({
                "Paziente": filename,
                "SSIM": SSIM,
                "PSNR": PSNR,
                "MAE": MAE
            })

            save_and_visualize_results(
                image,
                reconstruction,
                output_dir,
                filename=filename,
                index=i,
                batch=batch,
                save_nifti=True,
                save_png=False,
                visualize_every=1
            )


    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    #print(f"[INFO] Metriche salvate in {output_dir}/metrics.csv")


    print("\n--- Metriche Medie ---")
    print(f"SSIM: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    print(f"PSNR: {df['PSNR'].mean():.2f} ± {df['PSNR'].std():.2f}")
    print(f"MAE:  {df['MAE'].mean():.6f} ± {df['MAE'].std():.6f}")




if __name__ == "__main__":
    opt = TestOptions()
    test_autoencoder(opt)


