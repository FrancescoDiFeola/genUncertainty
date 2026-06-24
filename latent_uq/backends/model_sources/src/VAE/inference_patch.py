import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
from pathlib import Path

from torch.cuda.amp import autocast
from skimage.metrics import structural_similarity as ssim
from monai.metrics import PSNRMetric, SSIMMetric

from configs.test_options import TestOptions
from models.autoencoder import Autoencoder
from data.dataset_PATCH import CreateDataloader
from src.VAE.utils.checkpoints_utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def compute_metrics(image, reconstruction):
    image_np = image.squeeze().cpu().float().numpy()
    reconstruction_np = reconstruction.squeeze().cpu().float().numpy()

    #print(f"Min valore input: {np.min(img_fu_np).item():.4f}, Max valore input: {np.max(img_fu_np).item():.4f}")
    #print(f"Min valore input: {np.min(recon_fu_np).item():.4f}, Max valore input: {np.max(recon_fu_np).item():.4f}")

    #data_range = img_fu_np.max() - img_fu_np.min()
    data_range = 1.0

    ssim_calc = SSIMMetric(spatial_dims=3, data_range=data_range)
    ssim_val = ssim_calc(reconstruction, image)
    SSIM = ssim_val.item()
    ssim_calc.reset()

    psnr_calc = PSNRMetric(max_val=1.0)
    psnr_val = psnr_calc(reconstruction, image)
    PSNR = psnr_val.item()
    psnr_calc.reset()
    #SSIM = ssim(image_np, reconstruction_np, data_range=data_range, channel_axis=None)
    #PSNR = psnr(image_np, reconstruction_np, data_range=data_range)
    MAE = np.mean(np.abs(image_np - reconstruction_np))

    return SSIM, PSNR, MAE



def compute_metrics_masked(image, reconstruction, mask):
    image_np = image.squeeze().cpu().float().numpy()
    reconstruction_np = reconstruction.squeeze().cpu().float().numpy()
    mask_np = mask.squeeze().cpu().float().numpy()


    mask_np = (mask_np > 0).astype(float)

    # Calcola il numero di pixel validi (area della lesione)
    # Aggiungiamo un epsilon per evitare divisioni per zero se non c'è maschera
    num_pixels = np.sum(mask_np)
    if num_pixels == 0:
        return 0.0, 0.0, 0.0

    data_range = 1.0

    diff = np.abs(image_np - reconstruction_np)
    MAE = np.mean(diff * mask_np) / num_pixels

    mse = np.sum((image_np - reconstruction_np) ** 2 * mask_np) / num_pixels
    if mse == 0:
        PSNR = 100.0
    else:
        PSNR = 10 * np.log10((data_range ** 2) / mse)

    # --- CALCOLO SSIM ---
    # Usiamo full=True per ottenere la mappa dell'immagine (uguale dimensione dell'input)
    # win_size deve essere dispari e minore delle dimensioni dell'immagine (es. 7 o 11)
    ssim_val, ssim_map = ssim(
        image_np,
        reconstruction_np,
        data_range=data_range,
        full=True,  # IMPORTANTE: ci restituisce la mappa
        #win_size=7  # Finestra standard, adattabile
        channel_axis = None
    )

    SSIM = np.sum(ssim_map * mask_np) / num_pixels

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



def test_autoencoder(opt):

    output_path = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/src/BASELINE/VAE"
    output_dir = os.path.join(output_path, "inference_prova")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading dataset...")
    test_loader = CreateDataloader(opt, shuffle=True)

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
    )#.to(device)


    checkpoint_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/src/BASELINE/VAE/checkpoints_prova"
    _ = load_checkpoint(autoencoder, optimizer=None, checkpoint_dir=checkpoint_dir, model_name="autoencoder")
    #print("[INFO] Pesi caricati con successo.")

    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    metrics_list = []
    metrics_list_masked = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inferenza")):

            image = batch['image'].to(device)
            mask = batch['mask'].to(device)

            raw_filename = batch['patient_id']
            filename = raw_filename[0] if isinstance(raw_filename, list) else raw_filename

            raw_class = batch['class_name']
            lesion_class = raw_class[0] if isinstance(raw_class, list) else raw_class

            with autocast(enabled=True, dtype=torch.bfloat16):
                outputs = autoencoder(image)
                reconstruction = outputs[-1]



            SSIM, PSNR, MAE = compute_metrics(image, reconstruction)
            metrics_list.append({
                "Paziente": filename,
                "Classe": lesion_class,
                "SSIM": SSIM,
                "PSNR": PSNR,
                "MAE": MAE
            })


            SSIM, PSNR, MAE = compute_metrics_masked(image, reconstruction, mask)
            metrics_list_masked.append({
                "Paziente": filename,
                "Classe": lesion_class,
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
                save_nifti=False,
                save_png=False,
                visualize_every=25
            )


    df = pd.DataFrame(metrics_list)
    #df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


    df_mask = pd.DataFrame(metrics_list_masked)
    #df_mask.to_csv(os.path.join(output_dir, "metrics_masked.csv"), index=False)


    print("\n--- Metriche Medie ---")
    print(f"PSNR: {df['PSNR'].mean():.4f} ± {df['PSNR'].std():.4f}")
    print(f"SSIM: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    print(f"MAE:  {df['MAE'].mean():.4f} ± {df['MAE'].std():.4f}")




if __name__ == "__main__":
    opt = TestOptions()
    test_autoencoder(opt)


