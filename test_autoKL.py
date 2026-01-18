import os
import argparse
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from tqdm import tqdm
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from generative.losses import PerceptualLoss
from src import CT2DSliceDifferenceDataset
from src import init_autoencoder

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def denormalize(image, mean_HU=-1024, std_HU=600):
    """Convert normalized image back to original HU range."""
    return image * std_HU + mean_HU


def analyze_pixel_distribution(dataloader, model, device, output_dir, bins=100):
    """
    Compute pixel value distributions efficiently using incremental histogram updates.
    Avoids memory overflow by not storing all pixel values.
    """

    model.eval()

    # Initialize histograms with fixed bins
    bin_edges = np.linspace(-50, 50, bins + 1)  # Fixed bin edges in HU range
    hist_original = np.zeros(bins, dtype=np.float64)
    hist_reconstructed = np.zeros(bins, dtype=np.float64)

    # Running statistics (mean and variance)
    count = 0
    mean_original, mean_reconstructed = 0.0, 0.0
    m2_original, m2_reconstructed = 0.0, 0.0  # For variance calculation

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing Pixel Distribution"):
            diff_slices = batch["difference"].to(device)
            mean_D = batch["mean_D"].cpu().numpy()
            std_D = batch["std_D"].cpu().numpy()

            # Forward pass: Get reconstructed images
            reconstructed, _, _ = model(diff_slices)

            # Denormalize images back to original HU range
            diff_slices_denorm = denormalize(diff_slices.cpu().numpy(), mean_D, std_D)
            reconstructed_denorm = denormalize(reconstructed.cpu().numpy(), mean_D, std_D)

            # Update histograms (batch-wise)
            hist_original += np.histogram(diff_slices_denorm.flatten(), bins=bin_edges)[0]
            hist_reconstructed += np.histogram(reconstructed_denorm.flatten(), bins=bin_edges)[0]

            # Compute running mean and variance (Welford's algorithm)
            batch_size = diff_slices_denorm.size
            count += batch_size

            delta_original = diff_slices_denorm.mean() - mean_original
            mean_original += delta_original * batch_size / count
            m2_original += np.sum((diff_slices_denorm - mean_original) ** 2)

            delta_reconstructed = reconstructed_denorm.mean() - mean_reconstructed
            mean_reconstructed += delta_reconstructed * batch_size / count
            m2_reconstructed += np.sum((reconstructed_denorm - mean_reconstructed) ** 2)

    # Compute final standard deviation
    std_original = np.sqrt(m2_original / count) if count > 1 else 0
    std_reconstructed = np.sqrt(m2_reconstructed / count) if count > 1 else 0

    print(f"Original Mean: {mean_original:.2f}, Std: {std_original:.2f}")
    print(f"Reconstructed Mean: {mean_reconstructed:.2f}, Std: {std_reconstructed:.2f}")

    # Normalize histograms
    hist_original /= hist_original.sum()
    hist_reconstructed /= hist_reconstructed.sum()

    # Plot Histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], hist_original, width=np.diff(bin_edges), alpha=0.5, label="Original", color='blue', align="edge")
    plt.bar(bin_edges[:-1], hist_reconstructed, width=np.diff(bin_edges), alpha=0.5, label="Reconstructed", color='red', align="edge")
    plt.xlabel("HU Value")
    plt.ylabel("Normalized Frequency")
    plt.title("Overall Histogram of Pixel Intensities")
    plt.legend()
    plt.savefig(f"{output_dir}/HU_hist_distribution_autoKL_VT.png")
    plt.close()

    # Plot KDE (Kernel Density Estimation) - Approximating the distribution
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=bin_edges[:-1], y=hist_original, label="Original", color='blue')
    sns.lineplot(x=bin_edges[:-1], y=hist_reconstructed, label="Reconstructed", color='red')
    plt.xlabel("HU Value")
    plt.ylabel("Normalized Density")
    plt.title("Overall KDE of Pixel Intensities (Approximate)")
    plt.legend()
    plt.savefig(f"{output_dir}/HU_KDE_distribution_autoKL_VT.png")
    plt.close()


def compute_metrics(dataloader, model, device, output_csv):
    """Compute PSNR, SSIM, and MSE for the dataset and save to CSV."""
    results = []

    model.eval()

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Metrics"):
            ground_truth_img = batch["scan_1999"].to(device)  # "difference"
            reconstruction, _, _ = model(ground_truth_img)

            ground_truth_img = ground_truth_img.squeeze().cpu().numpy()
            reconstruction = reconstruction.squeeze().cpu().numpy()

            # Compute PSNR, SSIM, and MSE
            psnr = compute_psnr(ground_truth_img, reconstruction, data_range=ground_truth_img.max() - ground_truth_img.min())
            ssim = compute_ssim(ground_truth_img, reconstruction, data_range=ground_truth_img.max() - ground_truth_img.min())
            mse = np.mean((ground_truth_img - reconstruction) ** 2)

            # Append results
            results.append({'Image_Index': step, 'PSNR': psnr, 'SSIM': ssim, 'MSE': mse})

    # Save to CSV
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ['Image_Index', 'PSNR', 'SSIM', 'MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Metrics saved to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt', default=None, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_batch_size', default=1, type=int)
    parser.add_argument('--epoch_checkpoint', default=1000, type=int)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--analyze_distribution', action='store_true', help="Enable pixel intensity analysis")
    parser.add_argument('--compute_metrics', action='store_true', help="Enable PSNR, SSIM, and MSE computation")

    args = parser.parse_args()


    # Load the LDCT/HDCT dataset
    dataset = CT2DSliceDifferenceDataset(csv_file="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/virtual_treatment_NLST.csv", global_normalization=True)

    test_loader = DataLoader(dataset=dataset,
                             batch_size=args.max_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             persistent_workers=True,
                             pin_memory=True)

    # Initialize autoencoder
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    autoencoder.eval()
    # Compute PSNR, SSIM, and MSE if enabled
    if args.compute_metrics:
        output_csv = os.path.join(args.output_dir, f"metrics_test_{args.experiment_name}_{args.epoch_checkpoint}.csv")
        compute_metrics(test_loader, autoencoder, DEVICE, output_csv)

    # Perform pixel intensity analysis if enabled
    if args.analyze_distribution:
        analyze_pixel_distribution(test_loader, autoencoder, DEVICE, args.output_dir)
