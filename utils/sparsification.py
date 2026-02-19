import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------
# Helper: Compute AUSE and AURG
# --------------------------------------------
def compute_ause(fractions, curve):
    return np.trapz(curve, fractions)

def compute_aurg(fractions, curve, rand_curve):
    gap = rand_curve - curve
    return np.trapz(gap, fractions)

# --------------------------------------------
# Helper: Load and average sparsification CSV
# --------------------------------------------
def load_and_average(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure equal length (truncate if needed)
    df = df.iloc[:360201+1]

    grouped = df.groupby("Fraction").mean().reset_index()

    fractions = grouped["Fraction"].values
    error = grouped["Error"].values
    random_error = grouped["RandomError"].values
    oracle_error = grouped["OracleError"].values

    return fractions, error, random_error, oracle_error


# --------------------------------------------
# Plot + Compute metrics
# --------------------------------------------
def analyze_backbone(backbone_name, ours_csv, mc_csv, save_plot=True):

    f_o, err_o, rand_o, oracle_o = load_and_average(ours_csv)
    f_mc, err_mc, rand_mc, oracle_mc = load_and_average(mc_csv)

    # Use oracle/random from ours
    oracle = oracle_o
    random = rand_o

    # ----------------------------------------
    # Compute metrics
    # ----------------------------------------
    ause_ours = compute_ause(f_o, err_o)
    ause_mc = compute_ause(f_mc, err_mc)
    ause_random = compute_ause(f_o, random)
    ause_oracle = compute_ause(f_o, oracle)

    aurg_ours = compute_aurg(f_o, err_o, random)
    aurg_mc = compute_aurg(f_mc, err_mc, random)

    # Oracle-normalized AUSE
    norm_ause_ours = (ause_ours - ause_oracle) / (ause_random - ause_oracle + 1e-12)
    norm_ause_mc = (ause_mc - ause_oracle) / (ause_random - ause_oracle + 1e-12)

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    plt.figure(figsize=(6,5))

    plt.plot(f_o, err_o, linewidth=3, label="REFINE (Ours)")
    plt.plot(f_mc, err_mc, linewidth=2, label="MC-sampling")
    plt.plot(f_o, random, linestyle="--", linewidth=2, label="Random")
    plt.plot(f_o, oracle, linestyle=":", linewidth=2, label="Oracle")

    plt.xlabel("Fraction of removed pixels")
    plt.ylabel("Remaining normalized L1 error")
    plt.title(f"Sparsification - {backbone_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_plot:
        plt.savefig(f"{root}/{backbone_name}_sparsification.png", dpi=300)
    else:
        plt.show()

    plt.close()

    # ----------------------------------------
    # Return metrics
    # ----------------------------------------
    return {
        "Backbone": backbone_name,
        "AUSE_Ours": ause_ours,
        "AUSE_MC": ause_mc,
        "AUSE_Random": ause_random,
        "AUSE_Oracle": ause_oracle,
        "AURG_Ours": aurg_ours,
        "AURG_MC": aurg_mc,
        "NormAUSE_Ours": norm_ause_ours,
        "NormAUSE_MC": norm_ause_mc,
    }


# --------------------------------------------
# Run analysis for all models
# --------------------------------------------
root = "/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/sparsification"

models = {
    "DDPM": (
        f"{root}/DDPM_self_refining_sparsification_epoch_300.csv",
        f"{root}/DDPM_MC_sampling_sparsification_epoch_300.csv"
    ),
    "LDM": (
        f"{root}/LDM_self_refining_sparsification_epoch_50.csv",
        f"{root}/LDM_MC_sampling_sparsification_epoch_50.csv"
    ),
    "LFM": (
        f"{root}/LFM_self_refining_sparsification_K_30_epoch_250.csv",
        f"{root}/LFM_MC_sampling_sparsification_epoch_250.csv"
    ),
    "RF": (
        f"{root}/RF_self_refining_sparsification_K_30_epoch_50.csv",
        f"{root}/RF_MC_sampling_sparsification_epoch_50.csv"
    ),
}

results = []

for name, (ours, mc) in models.items():
    metrics = analyze_backbone(name, ours, mc)
    results.append(metrics)

# Save summary CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f"{root}/sparsification_summary_metrics.csv", index=False)

print(results_df)


##################

import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# Config
# --------------------------------------------------
MAX_ROW = 360201        # cut dataset consistently
NORMALIZE_TO_ORACLE = False   # optional normalization

# --------------------------------------------------
# Metric functions
# --------------------------------------------------

def compute_ause(fractions, curve):
    """Area Under Sparsification Error (lower is better)."""
    return np.trapz(curve, fractions)


def compute_aurg(fractions, curve, random_curve):
    """Area Under Random Gap (higher is better)."""
    gap = random_curve - curve
    return np.trapz(gap, fractions)


# --------------------------------------------------
# Per-sample metric computation
# --------------------------------------------------

def compute_metrics_per_sample(csv_path):
    df = pd.read_csv(csv_path)

    # Cut rows consistently
    df = df.iloc[:MAX_ROW+1]

    ause_list = []
    aurg_list = []

    for sample_id, sample_df in df.groupby("Sample"):

        fractions = sample_df["Fraction"].values
        error = sample_df["Error"].values
        random_error = sample_df["RandomError"].values
        oracle_error = sample_df["OracleError"].values

        ause = compute_ause(fractions, error)
        aurg = compute_aurg(fractions, error, random_error)

        # Optional normalization to oracle
        if NORMALIZE_TO_ORACLE:
            ause_oracle = compute_ause(fractions, oracle_error)
            ause_random = compute_ause(fractions, random_error)

            # Normalize between oracle and random
            ause = (ause - ause_oracle) / (ause_random - ause_oracle + 1e-8)

        ause_list.append(ause)
        aurg_list.append(aurg)

    ause_array = np.array(ause_list)
    aurg_array = np.array(aurg_list)

    results = {
        "AUSE_mean": ause_array.mean(),
        "AUSE_std": ause_array.std(),
        "AURG_mean": aurg_array.mean(),
        "AURG_std": aurg_array.std(),
        "num_samples": len(ause_array)
    }

    return results


# --------------------------------------------------
# Evaluate multiple models
# --------------------------------------------------

root = "/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/sparsification"

models = {
    "DDPM_Ours": f"{root}/DDPM_self_refining_sparsification_epoch_300.csv",
    "DDPM_MC": f"{root}/DDPM_MC_sampling_sparsification_epoch_300.csv",
    "LDM_Ours": f"{root}/LDM_self_refining_sparsification_epoch_50.csv",
    "LDM_MC": f"{root}/LDM_MC_sampling_sparsification_epoch_50.csv",
    "LFM_Ours": f"{root}/LFM_self_refining_sparsification_K_30_epoch_250.csv",
    "LFM_MC": f"{root}/LFM_MC_sampling_sparsification_epoch_250.csv",
    "RF_Ours": f"{root}/RF_self_refining_sparsification_K_30_epoch_50.csv",
    "RF_MC": f"{root}/RF_MC_sampling_sparsification_epoch_50.csv",
}

all_results = []

for name, path in models.items():
    print(f"Processing {name}...")
    metrics = compute_metrics_per_sample(path)
    metrics["Model"] = name
    all_results.append(metrics)

results_df = pd.DataFrame(all_results)

# Save results
results_df.to_csv(f"{root}/sparsification_metrics_summary.csv", index=False)

print("\nFinal Results:")
print(results_df)