import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# CONFIGURATION
# ======================================================

ROOT = "/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/sparsification"
COMMON_MAX_FRACTION = 0.9      # choose 0.8 or 0.9 consistently
SAVE_PLOTS = True

# ======================================================
# NUMERICAL INTEGRATION (SAFE)
# ======================================================

def integrate_curve(fractions, curve):
    order = np.argsort(fractions)
    fractions = fractions[order]
    curve = curve[order]

    # Remove duplicates
    fractions, unique_idx = np.unique(fractions, return_index=True)
    curve = curve[unique_idx]

    return np.trapz(curve, fractions)

# ======================================================
# CORRECT METRICS (FORMAL DEFINITIONS)
# ======================================================

def compute_ause(fractions, model_curve, oracle_curve):
    """
    AUSE = ∫ (model - oracle) / MAE(S) dα
    Lower is better. Oracle AUSE ≈ 0.
    """
    mae_S = model_curve[0]
    print(mae_S)
    spars_error = (model_curve - oracle_curve) / (mae_S + 1e-12)
    return integrate_curve(fractions, spars_error)

def compute_aurg(fractions, model_curve, random_curve):
    """
    AURG = ∫ (random - model) / MAE(S) dα
    Higher is better. Random AURG ≈ 0.
    """
    mae_S = model_curve[0]
    random_gap = (random_curve - model_curve) / (mae_S + 1e-12)
    return integrate_curve(fractions, random_gap)

# ======================================================
# LOAD CURVES (AVERAGED FOR PLOT)
# ======================================================

def load_average_curve(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["Fraction"] <= COMMON_MAX_FRACTION]

    grouped = (
        df.groupby("Fraction")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("Fraction")
    )

    return (
        grouped["Fraction"].values,
        grouped["Error"].values,
        grouped["RandomError"].values,
        grouped["OracleError"].values,
    )

# ======================================================
# PER-SAMPLE METRIC COMPUTATION
# ======================================================

def compute_metrics_per_sample(csv_path):

    df = pd.read_csv(csv_path)
    df = df[df["Fraction"] <= COMMON_MAX_FRACTION]

    ause_list = []
    aurg_list = []

    for sample_id, sample_df in df.groupby("Sample"):

        sample_df = sample_df.sort_values("Fraction")

        fractions = sample_df["Fraction"].values
        error = sample_df["Error"].values
        random_error = sample_df["RandomError"].values
        oracle_error = sample_df["OracleError"].values

        ause = compute_ause(fractions, error, oracle_error)
        aurg = compute_aurg(fractions, error, random_error)

        ause_list.append(ause)
        aurg_list.append(aurg)

    ause_array = np.array(ause_list)
    aurg_array = np.array(aurg_list)

    return {
        "AUSE_mean": ause_array.mean(),
        "AUSE_std": ause_array.std(ddof=1),
        "AURG_mean": aurg_array.mean(),
        "AURG_std": aurg_array.std(ddof=1),
        "num_samples": len(ause_array)
    }

# ======================================================
# PLOT SPARSIFICATION
# ======================================================

def plot_sparsification(backbone_name, ours_csv, mc_csv):

    f_o, err_o, rand_o, oracle_o = load_average_curve(ours_csv)
    f_mc, err_mc, _, _ = load_average_curve(mc_csv)

    plt.figure(figsize=(6,5))

    plt.plot(f_o, err_o, linewidth=3, label="REFINE (Ours)")
    plt.plot(f_mc, err_mc, linewidth=2, linestyle="--", label="MC-sampling")
    plt.plot(f_o, rand_o, linestyle="--", linewidth=2, label="Random")
    #  plt.plot(f_o, oracle_o, linestyle=":", linewidth=2, label="Oracle")

    plt.xlabel("Fraction of removed pixels")
    plt.ylabel("Remaining normalized L1 error")
    plt.title(f"Sparsification - {backbone_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(f"{ROOT}/{backbone_name}_sparsification.png", dpi=300)
    else:
        plt.show()

    plt.close()

# ======================================================
# RUN ANALYSIS
# ======================================================

models = {
    "DDPM": (
        f"{ROOT}/DDPM_self_refining_sparsification_epoch_300.csv",
        f"{ROOT}/DDPM_MC_sampling_sparsification_epoch_300.csv"
    ),
    "LDM": (
        f"{ROOT}/LDM_self_refining_sparsification_epoch_50.csv",
        f"{ROOT}/LDM_MC_sampling_sparsification_epoch_50.csv"
    ),
    "LFM": (
        f"{ROOT}/LFM_self_refining_sparsification_K_30_epoch_250.csv",
        f"{ROOT}/LFM_MC_sampling_sparsification_epoch_250.csv"
    ),
    "RF": (
        f"{ROOT}/RF_self_refining_sparsification_K_30_epoch_50.csv",
        f"{ROOT}/RF_MC_sampling_sparsification_epoch_50.csv"
    ),
}

all_results = []

for name, (ours_path, mc_path) in models.items():

    print(f"\nProcessing {name}...")

    metrics_ours = compute_metrics_per_sample(ours_path)
    metrics_mc = compute_metrics_per_sample(mc_path)

    metrics_ours["Model"] = f"{name}_Ours"
    metrics_mc["Model"] = f"{name}_MC"

    all_results.append(metrics_ours)
    all_results.append(metrics_mc)

    plot_sparsification(name, ours_path, mc_path)

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{ROOT}/sparsification_metrics_summary.csv", index=False)

print("\nFinal Results:")
print(results_df)