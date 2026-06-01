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

    elif writer_type == "metrics_no_uncertainty":
        csvfile= open(csv_path, mode='w', newline='')
        fieldnames = ['Sample', 'MSE', 'PSNR', 'SSIM']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        return (csvfile, writer)

    elif writer_type == "uncertainty_eval":
        csvfile= open(csv_path, mode='w', newline='')
        fieldnames = ['Sample', 'MAE', 'u_mean', 'u_p95', 'u_p99', 'u_top1_mean', 'top5_u_mean']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        return (csvfile, writer)

    elif writer_type == "uncertainty_cal":
        csvfile= open(csv_path, mode='w', newline='')
        fieldnames = ["sample", "bin", "p_low", "p_high", "mean_uncertainty", "mean_error", "count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        return (csvfile, writer)

    elif writer_type == "calibration":

        csvfile_2 = open(csv_path_2, mode='w', newline='')
        fieldnames_2 = ['Sample', 'Bin', 'Unc_mean', 'Err_mean', 'Count', 'Type']

        writer_2 = csv.DictWriter(csvfile_2, fieldnames=fieldnames_2)
        writer_2.writeheader()

        return (csvfile_2, writer_2)


    elif writer_type == "sparsification":

        csvfile = open(csv_path, mode='w', newline='')
        fieldnames = ['Sample', 'Fraction', 'Error', 'RandomError', 'OracleError']

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

def sparsification_curve(u, e, num_steps=50, max_frac=0.99):
    """
    u: flattened uncertainty
    e: flattened error
    """

    sorted_idx = np.argsort(-u) # ordine degli indici dei pixel a incertezza decrescente
    e_sorted = e[sorted_idx]  # ordino l'errore in base all'ordine definito sopra

    N = len(e_sorted)  # lunghezza del vettore

    # Avoid alpha = 1
    fractions = np.linspace(0, max_frac, num_steps)
    curve = []

    for frac in fractions:
        k = int(frac * N)

        if k >= N:
            curve.append(0.0)
        else:
            remaining_error = e_sorted[k:]  # Remove top k pixels (quelli che corrispondono ad icnertezza maggiore in base alla mappa di incetezza)
            curve.append(remaining_error.mean()) # Compute mean error on remaining pixels

    curve = np.array(curve)

    # Normalize by initial error
    curve = curve / curve[0]

    return fractions[:len(curve)], np.array(curve)

def sparsification_curve_fast(u, e, num_steps=50, max_frac=0.99, normalize=True):

    sorted_idx = np.argsort(-u)
    e_sorted = e[sorted_idx]

    N = len(e_sorted)

    fractions = np.linspace(0, max_frac, num_steps)
    k_vals = np.minimum(np.round(fractions * N).astype(int), N - 1)

    cumsum = np.cumsum(e_sorted)
    total_sum = cumsum[-1]

    remaining_sum = np.where(
        k_vals == 0,
        total_sum,
        total_sum - cumsum[k_vals - 1]
    )

    remaining_count = N - k_vals
    curve = remaining_sum / remaining_count

    if normalize:
        curve /= (curve[0] + 1e-12)

    return fractions, curve

def uncertainty_error_tail_bins_torch(
    uncertainty,
    error,
    sample_id,
    percentiles=(0, 50, 75, 90, 95, 99, 100),
):
    u = np.asarray(uncertainty).reshape(-1)
    e = np.asarray(error).reshape(-1)
    valid = np.isfinite(u) & np.isfinite(e)
    u = u[valid]
    e = e[valid]
    edges = np.percentile(u, percentiles)
    rows = []
    for b in range(len(edges) - 1):
        if b == len(edges) - 2:
            idx = (u >= edges[b]) & (u <= edges[b + 1])
        else:
            idx = (u >= edges[b]) & (u < edges[b + 1])
        n = int(idx.sum())
        if n == 0:
            continue
        rows.append({
            "sample": sample_id,
            "bin": b,
            "p_low": percentiles[b],
            "p_high": percentiles[b + 1],
            "mean_uncertainty": float(u[idx].mean()),
            "mean_error": float(e[idx].mean()),
            "count": n,
        })
    return rows

def random_sparsification_fast(e, fractions, trials=20, normalize=True):
    """
    Vectorized random sparsification curve.

    e: flattened error (1D numpy array)
    fractions: array of removal fractions (same as sparsification_curve)
    trials: number of random permutations
    """

    N = len(e)
    avg_curve = np.zeros(len(fractions), dtype=np.float64)

    # Precompute k values once
    k_vals = np.minimum(np.round(fractions * N).astype(int), N - 1)

    for _ in range(trials):

        perm = np.random.permutation(N)
        e_perm = e[perm]

        cumsum = np.cumsum(e_perm)
        total_sum = cumsum[-1]

        # Remaining sum using safe formulation
        remaining_sum = np.where(
            k_vals == 0,
            total_sum,
            total_sum - cumsum[k_vals - 1]
        )

        remaining_count = N - k_vals
        avg_curve += remaining_sum / remaining_count

    avg_curve /= trials

    if normalize:
        avg_curve /= (avg_curve[0] + 1e-12)

    return avg_curve

def random_sparsification(e, fractions, trials=10):
    N = len(e)
    avg_curve = np.zeros(len(fractions))

    for _ in range(trials):
        perm = np.random.permutation(N)
        e_perm = e[perm]

        tmp = []
        for frac in fractions:
            k = int(frac * N)
            tmp.append(e_perm[k:].mean())

        avg_curve += np.array(tmp)

    avg_curve /= trials

    # Normalize
    avg_curve /= avg_curve[0]

    return avg_curve


def compute_ause(fractions, curve):
    """
    Normalized Area Under Sparsification Error (lower is better).
    Fractions must be increasing and start at 0.
    """
    area = np.trapz(curve, fractions)
    f_max = fractions[-1]
    return area / f_max


def compute_aurg(fractions, curve, rand_curve):
    """
    Normalized Area Under Random Gap (higher is better).
    """
    gap = rand_curve - curve
    area = np.trapz(gap, fractions)
    f_max = fractions[-1]
    return area / f_max

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

def summarize_uncertainty(U):

    """

    Compute compact statistics from an uncertainty map.

    Parameters

    ----------

    U : np.ndarray or torch.Tensor

        Uncertainty map of shape:

        - (H, W)

        - (1, H, W)

        - (C, H, W)

    Returns

    -------

    dict

        {

            "u_mean",

            "u_p95",

            "u_p99",

            "u_top1_mean",

            "u_top5_mean"

        }

    """

    # convert torch -> numpy if needed

    if hasattr(U, "detach"):

        U = U.detach().cpu().numpy()

    U = np.asarray(U).astype(np.float64)

    # flatten

    u = U.reshape(-1)

    # percentiles

    u_p95 = np.percentile(u, 95)

    u_p99 = np.percentile(u, 99)

    # top-k means

    n = len(u)

    top1_k = max(1, int(0.01 * n))

    top5_k = max(1, int(0.05 * n))

    u_sorted = np.sort(u)

    u_top1_mean = np.mean(u_sorted[-top1_k:])

    u_top5_mean = np.mean(u_sorted[-top5_k:])

    return {

        "u_mean": float(np.mean(u)),

        "u_p95": float(u_p95),

        "u_p99": float(u_p99),

        "u_top1_mean": float(u_top1_mean),

        "u_top5_mean": float(u_top5_mean),

    }