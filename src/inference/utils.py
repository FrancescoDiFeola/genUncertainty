import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
import csv

def initialize_writers(
    csv_path=None,
    csv_path_2=None,
    writer_type="both"
):
    """
    Initialize CSV writer(s).

    Args:
        csv_path: path for metrics or custom CSV
        csv_path_2: path for calibration CSV
        writer_type: one of
            - "metrics"
            - "calibration"
            - "refine"
            - "both"

    Returns:
        Depending on writer_type:
            metrics        -> csvfile, writer
            calibration    -> csvfile_2, writer_2
            refine         -> csvfile, writer
            both           -> csvfile, csvfile_2, writer, writer_2
    """

    if writer_type == "metrics":

        csvfile = open(csv_path, mode='w', newline='')
        fieldnames = [
            'Sample', 'MSE', 'PSNR', 'SSIM',
            'Pearson_u_norm', 'Spearman_u_norm',
            'AUROC_top15_u_norm', 'AUROC_top10_u_norm', 'AUROC_top5_u_norm',
            'Pearson_u_unnorm', 'Spearman_u_unnorm',
            'AUROC_top15_u_unnorm', 'AUROC_top10_u_unnorm', 'AUROC_top5_u_unnorm'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        return (csvfile, writer)


    elif writer_type == "calibration":

        csvfile_2 = open(csv_path_2, mode='w', newline='')
        fieldnames_2 = ['Sample', 'Bin', 'Unc_mean', 'Err_mean', 'Count', 'Type']

        writer_2 = csv.DictWriter(csvfile_2, fieldnames=fieldnames_2)
        writer_2.writeheader()

        return (csvfile_2, writer_2)


    elif writer_type == "sparrsification":

        csvfile = open(csv_path, mode='w', newline='')
        fieldnames = ['Sample', 'Method', 'Fraction', 'Error', 'RandomError']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        return (csvfile, writer)


    elif writer_type == "both":

        csvfile = open(csv_path, mode='w', newline='')
        csvfile_2 = open(csv_path_2, mode='w', newline='')

        fieldnames = [
            'Sample', 'MSE', 'PSNR', 'SSIM',
            'Pearson_u_norm', 'Spearman_u_norm',
            'AUROC_top15_u_norm', 'AUROC_top10_u_norm', 'AUROC_top5_u_norm',
            'Pearson_u_unnorm', 'Spearman_u_unnorm',
            'AUROC_top15_u_unnorm', 'AUROC_top10_u_unnorm', 'AUROC_top5_u_unnorm'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        fieldnames_2 = ['Sample', 'Bin', 'Unc_mean', 'Err_mean', 'Count', 'Type']
        writer_2 = csv.DictWriter(csvfile_2, fieldnames=fieldnames_2)
        writer_2.writeheader()

        return (csvfile, csvfile_2, writer, writer_2)


    else:
        raise ValueError(f"Unknown writer_type: {writer_type}")

def sparsification_curve(u, e, num_steps=50):
    sorted_idx = np.argsort(-u) # ordine degli indici dei pixel a incertezza decrescente
    e_sorted = e[sorted_idx]  # ordino l'errore in base all'ordine definito sopra

    N = len(e_sorted)  # lunghezza del vettore
    fractions = np.linspace(0, 1, num_steps)
    curve = []

    for frac in fractions:
        k = int(frac * N)

        if k >= N:
            curve.append(0.0)
        else:
            remaining_error = e_sorted[k:]  # Remove top k pixels (quelli che corrispondono ad icnertezza maggiore in base alla mappa di incetezza)
            curve.append(remaining_error.mean()) # Compute mean error on remaining pixels

    return fractions, np.array(curve)

def random_sparsification(e, fractions, trials=5):
    N = len(e)
    avg_curve = np.zeros(len(fractions))
    # Random ranking baseline  (l'ordinamento dei pixel non è più basato sulla mappa di incertezza ma è randomico
    for _ in range(trials):  # medio su più tentativi
        perm = np.random.permutation(N)
        e_perm = e[perm]

        tmp = []
        for frac in fractions:
            k = int(frac * N)
            if k >= N:
                tmp.append(0.0)
            else:
                tmp.append(e_perm[k:].mean())

        avg_curve += np.array(tmp)

    return avg_curve / trials

def compute_ause(fractions, curve):
    """Area Under Sparsification Error (lower is better); it measures total residual error"""
    return np.trapz(curve, fractions)

def compute_aurg(fractions, curve, rand_curve):
    """Area Under Random Gap (higher is better); AURG measures gain over random"""
    gap = rand_curve - curve
    return np.trapz(gap, fractions)

def map_correlations_multi_thresholds(unc_map, pred, gt, percentiles=(95, 90, 85)):
    """
    Compute correlation and failure-discrimination metrics between uncertainty and error maps.

    For each percentile p:
      - define failure pixels as top (100 - p)% highest-error pixels
      - compute AUROC of uncertainty predicting failure

    Args:
        unc_map (np.ndarray): uncertainty map (H, W) or (C, H, W)
        pred (np.ndarray): prediction
        gt (np.ndarray): ground truth
        percentiles (tuple): percentiles defining error thresholds
                             (95 -> top 5%, 90 -> top 10%, etc.)

    Returns:
        results (dict): dictionary with:
            - pearson
            - spearman
            - auroc_top_5
            - auroc_top_10
            - auroc_top_15
    """

    # --- error map ---
    err = np.abs(pred - gt)

    # flatten
    u = unc_map.flatten()
    e = err.flatten()

    # remove NaN / Inf
    mask = np.isfinite(u) & np.isfinite(e)
    u = u[mask]
    e = e[mask]

    results = {}

    # --- global correlations ---
    results["pearson"] = pearsonr(u, e)[0]
    results["spearman"] = spearmanr(u, e)[0]

    # --- failure discrimination at multiple thresholds ---
    for p in percentiles:
        err_thresh = np.percentile(e, p)
        err_bin = (e > err_thresh).astype(np.int32)

        # AUROC is only valid if both classes exist
        if len(np.unique(err_bin)) > 1:
            auroc = roc_auc_score(err_bin, u)
        else:
            auroc = np.nan

        results[f"AUROC_top{100-p}"] = auroc

    return results

def map_correlations(unc_map, pred, gt):
    """
    This function computes the pixel-wise correlation between the model’s uncertainty map
    and the true reconstruction error. The uncertainty map is normalized on a per-sample basis
    (norm_percentile) to calibrate differences in global scale across images, while the error
    map is kept in its raw physical units. Per-image normalization preserves the spatial pattern
    of uncertainty (relative high/low values) and makes maps comparable across the dataset
    without distorting the true error magnitude. This provides a meaningful assessment of how
    well uncertainty predicts local reconstruction inaccuracies.
    """
    # error map
    err = np.abs(pred - gt)

    # flatten for correlation
    u = unc_map.flatten()
    e = err.flatten()

    # remove NaN/inf
    mask = np.isfinite(u) & np.isfinite(e)
    u = u[mask]
    e = e[mask]

    pear = pearsonr(u, e)[0]
    spear = spearmanr(u, e)[0]

    return pear, spear, err

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
def collect_calibration_data(
    unc_map: torch.Tensor,
    err_map: torch.Tensor,
    num_bins: int = 15,  # number of callibration bins
):
    """
    Collect per-bin statistics for calibration (ECE / reliability).

    Args:
        unc_map: (H, W) uncertainty (std or variance, NOT normalized)
        err_map: (H, W) absolute error
    Returns:
        bin_unc_mean, bin_err_mean, bin_count
    """
    u = unc_map.flatten()
    e = err_map.flatten()

    mask = torch.isfinite(u) & torch.isfinite(e)
    u = u[mask]
    e = e[mask]

    # Define bins over uncertainty  [q1, q2, ...., q15], Each bin contains roughly the same number of pixels, bobust to heavy-tailed uncertainty distributions, Prevents empty bins, standard in the literature
    bins = torch.quantile(u, torch.linspace(0, 1, num_bins + 1, device=u.device))
    bin_ids = torch.bucketize(u, bins[1:-1]) # assign each pixel to a bin

    bin_unc_mean = []
    bin_err_mean = []
    bin_count = []

    for b in range(num_bins):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        bin_unc_mean.append(u[idx].mean().item()) # map the predicted uncertaintu in the bin, Average model-predicted uncertainty for pixels in bin b.
        bin_err_mean.append(e[idx].mean().item()) # Average actual reconstruction error for the same pixels.
        bin_count.append(idx.sum().item())

    return bin_unc_mean, bin_err_mean, bin_count
