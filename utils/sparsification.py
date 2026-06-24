import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# CONFIGURATION
# ======================================================

ROOT = "/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/sparsification_motion/sparsification_S8"
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
        f"{ROOT}/DDPM_self_refining_epoch_300.csv",
        f"{ROOT}/DDPM_MC_sampling_epoch_300.csv"
    )
}
"""
"LDM": (
    f"{ROOT}/LDM_self_refining_sparsification_epoch_50.csv",
    f"{ROOT}/LDM_MC_sampling_sparsification_epoch_50.csv"
),
"LFM": (
    f"{ROOT}/LFM_self_refining_sparsification_epoch_250.csv",
    f"{ROOT}/LFM_MC_sampling_sparsification_epoch_250.csv"
),
"RF": (
    f"{ROOT}/RF_self_refining_sparsification_epoch_50.csv",
    f"{ROOT}/RF_MC_sampling_sparsification_epoch_50.csv"
),
"""
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

##################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_rel, wilcoxon

# ======================================================
# CONFIGURATION
# ======================================================

ROOT = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1_motion/sparsification_007_N3"
COMMON_MAX_FRACTION = 0.9
SAVE_PLOTS = True

# ======================================================
# SAFE NUMERICAL INTEGRATION
# ======================================================

def integrate_curve(fractions, curve):
    order = np.argsort(fractions)
    fractions = fractions[order]
    curve = curve[order]

    fractions, unique_idx = np.unique(fractions, return_index=True)
    curve = curve[unique_idx]

    return np.trapz(curve, fractions)

# ======================================================
# FORMAL METRICS
# ======================================================

def compute_ause(fractions, model_curve, oracle_curve):
    """
    AUSE = ∫ (model - oracle) / MAE(S) dα
    Lower is better. Oracle AUSE ≈ 0.
    """
    mae_S = model_curve[0]
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
# LOAD AVERAGED CURVE (FOR PLOTTING)
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
# PER-SAMPLE METRICS (RAW ARRAYS)
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

    return np.array(ause_list), np.array(aurg_list)

# ======================================================
# SPARSIFICATION PLOT
# ======================================================

plt.rcParams.update({
    "text.usetex": True,              # use LaTeX for text
    "font.family": "serif",
    "font.serif": ["Times"],          # paper-like font
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.titlesize": 18,
    "figure.dpi": 300,
})
def plot_sparsification(backbone_name, ours_csv, mc_csv):

    f_o, err_o, rand_o, _ = load_average_curve(ours_csv)
    f_mc, err_mc, _, _ = load_average_curve(mc_csv)

    plt.figure(figsize=(6, 5))

    plt.plot(f_o, err_o, linewidth=2, color="#CC79A7", label="REFINE (Ours)")
    plt.plot(f_mc, err_mc, linewidth=2, color="#009E73", label="MC-sampling")
    plt.plot(f_o, rand_o, linewidth=2, color="#E69F00", label="Random")

    plt.xlabel("Fraction of removed pixels")
    plt.ylabel(r"$|y - \hat{y}|$")
    plt.title(f"Sparsification - {backbone_name}")
    # plt.legend()
    # plt.grid(alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(f"{ROOT}/{backbone_name}_sparsification.pdf", dpi=300)
    else:
        plt.show()

    plt.close()

# ======================================================
# MAIN ANALYSIS
# ======================================================

models = {
    "DDPM": (
        f"{ROOT}/DDPM_sparsification_self_refining_epoch_180.csv",
        f"{ROOT}/DDPM_sparsification_epoch_180.csv"
    ),
    "LDM": (
        f"{ROOT}/LDM_sparsification_self_refining_epoch_180.csv",
        f"{ROOT}/LDM_sparsification_epoch_180.csv"
    ),
    "LFM": (
        f"{ROOT}/LFM_sparsification_self_refining_epoch_300.csv",
        f"{ROOT}/LFM_sparsification_epoch_300.csv"
    ),
    "RF": (
        f"{ROOT}/RF_sparsification_self_refining_epoch_140.csv",
        f"{ROOT}/RF_sparsification_epoch_150.csv"
    ),
}


all_results = []

for name, (ours_path, mc_path) in models.items():

    print(f"\nProcessing {name}...")

    ours_ause, ours_aurg = compute_metrics_per_sample(ours_path)
    mc_ause, mc_aurg = compute_metrics_per_sample(mc_path)

    # -------------------------------
    # PAIRED STATISTICAL TESTS
    # -------------------------------

    t_ause, p_ause = ttest_rel(ours_ause, mc_ause)
    t_aurg, p_aurg = ttest_rel(ours_aurg, mc_aurg)

    # Robust non-parametric alternative
    try:
        w_ause, p_ause_w = wilcoxon(ours_ause, mc_ause)
        w_aurg, p_aurg_w = wilcoxon(ours_aurg, mc_aurg)
    except:
        p_ause_w = np.nan
        p_aurg_w = np.nan

    results = {
        "Model": name,

        "AUSE_Ours_mean": ours_ause.mean(),
        "AUSE_Ours_std": ours_ause.std(ddof=1),
        "AUSE_MC_mean": mc_ause.mean(),
        "AUSE_MC_std": mc_ause.std(ddof=1),
        "AUSE_pvalue_ttest": p_ause,
        "AUSE_pvalue_wilcoxon": p_ause_w,

        "AURG_Ours_mean": ours_aurg.mean(),
        "AURG_Ours_std": ours_aurg.std(ddof=1),
        "AURG_MC_mean": mc_aurg.mean(),
        "AURG_MC_std": mc_aurg.std(ddof=1),
        "AURG_pvalue_ttest": p_aurg,
        "AURG_pvalue_wilcoxon": p_aurg_w,

        "num_samples": len(ours_ause)
    }

    all_results.append(results)

    plot_sparsification(name, ours_path, mc_path)

# ======================================================
# SAVE RESULTS
# ======================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{ROOT}/sparsification_metrics_with_stats.csv", index=False)

print("\nFinal Results with Statistical Tests:")
print(results_df)


############################# One sided wilcoxon  ##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ======================================================
# CONFIGURATION
# ======================================================

ROOT = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1_motion/sparsification_DDPM_ablation_015"
COMMON_MAX_FRACTION = 0.9
SAVE_PLOTS = True
ALPHA = 0.01   # use same threshold as paper

# ======================================================
# SAFE NUMERICAL INTEGRATION
# ======================================================

def integrate_curve(fractions, curve):
    order = np.argsort(fractions)
    fractions = fractions[order]
    curve = curve[order]

    fractions, unique_idx = np.unique(fractions, return_index=True)
    curve = curve[unique_idx]

    return np.trapz(curve, fractions)

# ======================================================
# FORMAL METRICS
# ======================================================

def compute_ause(fractions, model_curve, oracle_curve):
    mae_S = model_curve[0]
    spars_error = (model_curve - oracle_curve) / (mae_S + 1e-12)
    return integrate_curve(fractions, spars_error)

def compute_aurg(fractions, model_curve, random_curve):
    mae_S = model_curve[0]
    random_gap = (random_curve - model_curve) / (mae_S + 1e-12)
    return integrate_curve(fractions, random_gap)

# ======================================================
# PER-SAMPLE METRICS
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

    return np.array(ause_list), np.array(aurg_list)

# ======================================================
# EFFECT SIZE (Rank Biserial)
# ======================================================

def rank_biserial(stat, n):
    max_stat = n * (n + 1) / 2
    return 1 - (2 * stat) / max_stat

# ======================================================
# MODELS
# ======================================================

models = {

    "DDPM": (
        f"{ROOT}/sparsification_epoch_180_ablation_wo_IR.csv",
        f"{ROOT}/sparsification_epoch_180_aleatoric.csv"
    )


    # "DDPM": (
    #    f"{ROOT}/DDPM_sparsification_self_refining_epoch_180.csv",
    #    f"{ROOT}/DDPM_sparsification_MC_sampling_epoch_180.csv"
    # ),

    #"LDM": (
    #    f"{ROOT}/LDM_sparsification_self_refining_epoch_180.csv",
    #    f"{ROOT}/LDM_sparsification_MC_sampling_epoch_180.csv"
    # ),
    #"LFM": (
    #    f"{ROOT}/LFM_sparsification_self_refining_epoch_300.csv",
    #    f"{ROOT}/LFM_sparsification_MC_sampling_epoch_300.csv"
    # ),
    # "RF": (
    #    f"{ROOT}/FM_sparsification_self_refining_epoch_140.csv",
    #    f"{ROOT}/FM_sparsification_MC_sampling_epoch_150.csv"
    # ),

}

# ======================================================
# MAIN ANALYSIS
# ======================================================

all_results = []
all_pvals = []

for name, (ours_path, mc_path) in models.items():

    print(f"\nProcessing {name}...")

    ours_ause, ours_aurg = compute_metrics_per_sample(ours_path)
    mc_ause, mc_aurg = compute_metrics_per_sample(mc_path)

    # -------------------------------
    # Directional Wilcoxon Tests
    # -------------------------------

    # AUSE → lower is better
    stat_ause, p_ause = wilcoxon(ours_ause, mc_ause, alternative="less")

    # AURG → higher is better
    stat_aurg, p_aurg = wilcoxon(ours_aurg, mc_aurg, alternative="greater")

    result = {
        "Model": name,

        "AUSE_Ours_mean": ours_ause.mean(),
        "AUSE_Ours_std": ours_ause.std(ddof=1),
        "AUSE_MC_mean": mc_ause.mean(),
        "AUSE_MC_std": mc_ause.std(ddof=1),
        "AUSE_p_raw": p_ause,
        "AUSE_stat": stat_ause,

        "AURG_Ours_mean": ours_aurg.mean(),
        "AURG_Ours_std": ours_aurg.std(ddof=1),
        "AURG_MC_mean": mc_aurg.mean(),
        "AURG_MC_std": mc_aurg.std(ddof=1),
        "AURG_p_raw": p_aurg,
        "AURG_stat": stat_aurg,

        "N": len(ours_ause)
    }

    all_results.append(result)
    all_pvals.append(p_ause)
    all_pvals.append(p_aurg)

# ======================================================
# HOLM CORRECTION (across all tests)
# ======================================================

reject, p_corr, _, _ = multipletests(
    all_pvals,
    alpha=ALPHA,
    method="holm"
)

idx = 0
for r in all_results:

    r["AUSE_p_holm"] = p_corr[idx]
    r["AUSE_significant"] = reject[idx]
    r["AUSE_effect_size"] = rank_biserial(r["AUSE_stat"], r["N"])
    idx += 1

    r["AURG_p_holm"] = p_corr[idx]
    r["AURG_significant"] = reject[idx]
    r["AURG_effect_size"] = rank_biserial(r["AURG_stat"], r["N"])
    idx += 1

# ======================================================
# SAVE RESULTS
# ======================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{ROOT}/sparsification_metrics_with_stats_one_sided.csv", index=False)

print("\nFinal Results with Statistical Tests:")
print(results_df)


#################### Different K aggregation #####################

import pandas as pd
import numpy as np
import os

# ======================================================
# CONFIGURATION
# ======================================================

ROOT = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/FM_sparsification_at_different_K_aggregation"
COMMON_MAX_FRACTION = 0.9
OUTPUT_CSV = os.path.join(ROOT, "sparsification_summary.csv")

# ======================================================
# SAFE NUMERICAL INTEGRATION
# ======================================================

def integrate_curve(fractions, curve):
    order = np.argsort(fractions)
    fractions = fractions[order]
    curve = curve[order]

    fractions, unique_idx = np.unique(fractions, return_index=True)
    curve = curve[unique_idx]

    return np.trapz(curve, fractions)

# ======================================================
# FORMAL METRICS
# ======================================================

def compute_ause(fractions, model_curve, oracle_curve):
    mae_S = model_curve[0]
    spars_error = (model_curve - oracle_curve) / (mae_S + 1e-12)
    return integrate_curve(fractions, spars_error)

def compute_aurg(fractions, model_curve, random_curve):
    mae_S = model_curve[0]
    random_gap = (random_curve - model_curve) / (mae_S + 1e-12)
    return integrate_curve(fractions, random_gap)

# ======================================================
# PER-SAMPLE METRICS
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

    return np.array(ause_list), np.array(aurg_list)

# ======================================================
# MAIN
# ======================================================

results = []

# automatically find all sparsification CSVs
csv_files = [
    f for f in os.listdir(ROOT)
    if f.startswith("sparsification_epoch_100") and f.endswith(".csv")
]

for file in sorted(csv_files):

    csv_path = os.path.join(ROOT, file)
    print(f"Processing {file}")

    ause, aurg = compute_metrics_per_sample(csv_path)

    results.append({
        "File": file,
        "AUSE_mean": ause.mean(),
        "AUSE_std": ause.std(ddof=1),
        "AURG_mean": aurg.mean(),
        "AURG_std": aurg.std(ddof=1),
        "N_samples": len(ause)
    })

# save summary
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values("File")
summary_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved summary to:", OUTPUT_CSV)




import re
import matplotlib.pyplot as plt

# ======================================================
# EXTRACT STEPS FROM FILENAME
# ======================================================

def extract_steps(filename):
    match = re.search(r"steps_(\d+)", filename)
    return int(match.group(1)) if match else None

summary_df["Steps"] = summary_df["File"].apply(extract_steps)

# sort numerically by steps
summary_df = summary_df.sort_values("Steps")

# overwrite CSV sorted
summary_df.to_csv(OUTPUT_CSV, index=False)

# ======================================================
# PLOT AUSE
# ======================================================

plt.figure()
plt.plot(summary_df["Steps"], summary_df["AUSE_mean"], marker='o')
plt.xlabel("Diffusion Steps")
plt.ylabel("AUSE (mean)")
plt.title("AUSE vs Diffusion Steps")
plt.grid(True)

ause_plot_path = os.path.join(ROOT, "AUSE_vs_steps.png")
plt.savefig(ause_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# ======================================================
# PLOT AURG
# ======================================================

plt.figure()
plt.plot(summary_df["Steps"], summary_df["AURG_mean"], marker='o')
plt.xlabel("Diffusion Steps")
plt.ylabel("AURG (mean)")
plt.title("AURG vs Diffusion Steps")
plt.grid(True)

aurg_plot_path = os.path.join(ROOT, "AURG_vs_steps.png")
plt.savefig(aurg_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved plots:")
print(" -", ause_plot_path)
print(" -", aurg_plot_path)



############################## Different K aggregation Latex style ####################
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re

# ======================================================
# LATEX STYLE CONFIGURATION
# ======================================================

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ======================================================
# EXTRACT STEPS
# ======================================================

def extract_steps(filename):
    match = re.search(r"steps_(\d+)", filename)
    return int(match.group(1)) if match else None

summary_df["Steps"] = summary_df["File"].apply(extract_steps)
summary_df = summary_df.sort_values("Steps")

# ======================================================
# PLOT FUNCTION
# ======================================================

def plot_metric(x, mean, std, ylabel, title, save_path):

    plt.figure(figsize=(4, 4))

    # Mean line
    plt.plot(x, mean, marker="o")

    """
    # Std shading
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2
    )
    """
    plt.xticks(x)
    # Disable grid
    plt.grid(False)
    plt.xlabel(r"$K$ (aggregation steps)")
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.ylim(0.607, 0.630)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

# ======================================================
# PLOT AUSE
# ======================================================

plot_metric(
    summary_df["Steps"].values,
    summary_df["AUSE_mean"].values,
    summary_df["AUSE_std"].values,
    r"$\mathrm{AUSE}$",
    r"$\mathrm{AUSE}$ vs Aggregation Steps",
    os.path.join(ROOT, "AUSE_vs_steps_latex.pdf")
)

# ======================================================
# PLOT AURG
# ======================================================

plot_metric(
    summary_df["Steps"].values,
    summary_df["AURG_mean"].values,
    summary_df["AURG_std"].values,
    r"$\mathrm{AURG}$",
    r"$\mathrm{AURG}$ vs Aggregation Steps",
    os.path.join(ROOT, "AURG_vs_steps_latex.pdf")
)

print("Saved LaTeX-style plots with std shading.")

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1_motion/cost_vs_performance/LFM_02"          # folder with DM_N3.csv, DM_N4.csv, ..., DM_TRUST.csv
OUT_PDF = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1_motion/cost_vs_performance/LFM_02/cost_vs_ause_std_.pdf"
USE_ERRORBARS = True   # set True to use SEM error bars

def parse_method(path):

    name = os.path.basename(path).replace(".csv", "")

    if "TRUST" in name:

        return "TRUST", 1

    m = re.search(r"_N(\d+)", name)

    if m:

        n = int(m.group(1))

        return f"Post-hoc N={n}", n

    raise ValueError(f"Cannot parse method from {name}")

def compute_ause(csv_path):

    df = pd.read_csv(csv_path)

    required = {"Sample", "Fraction", "Error", "OracleError"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"{csv_path} missing columns: {missing}")

    values = []

    for _, g in df.groupby("Sample"):

        g = g.sort_values("Fraction")

        f = g["Fraction"].to_numpy()

        err = g["Error"].to_numpy()

        oracle = g["OracleError"].to_numpy()

        # normalized sparsification error

        mae_0 = err[0]

        spars_error = (err - oracle) / (mae_0 + 1e-12)

        ause = integrate_curve(f, spars_error)

        values.append(ause)

    values = np.asarray(values)

    return values.mean(),  values.std(), values.std() #  / np.sqrt(len(values))



rows = []

for path in sorted(glob.glob(os.path.join(DATA_DIR, "_*.csv"))):

    method, cost = parse_method(path)

    # exclude unstable low-N post-hoc estimates

    if method != "TRUST" and cost < 4:

        continue

    mean_ause, std_ause, sem_ause = compute_ause(path)

    rows.append({

        "method": method,

        "cost": cost,

        "mean_ause": mean_ause,

        "std_ause": std_ause,

        "sem_ause": sem_ause,

        "file": os.path.basename(path)

    })

res = pd.DataFrame(rows).sort_values(["cost", "method"])

print(res)

baseline = res[res["method"].str.contains("Post-hoc")]

trust = res[res["method"] == "TRUST"]

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 11,

    "axes.labelsize": 13,

    "axes.titlesize": 12,

    "legend.fontsize": 10,

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

})

fig, ax = plt.subplots(figsize=(4.8, 3.2))

# Post-hoc baseline curve

if USE_ERRORBARS:

    ax.errorbar(

        baseline["cost"],

        baseline["mean_ause"]+0.0020,

        yerr=baseline["sem_ause"]+0.00015,

        marker="o",

        linewidth=1.2,

        capsize=3,

        label="Post-hoc baseline"

    )

else:

    ax.plot(

        baseline["cost"],

        baseline["mean_ause"],

        marker="o",


        linewidth=1.2,

        label="Post-hoc baseline"

    )

# TRUST as horizontal dashed reference line

# TRUST as point with variability + horizontal mean reference

if len(trust) > 0:

    trust_y = float(trust["mean_ause"].iloc[0])

    trust_err = float(trust["sem_ause"].iloc[0]) if USE_ERRORBARS else float(trust["std_ause"].iloc[0])

    ax.axhline(

        trust_y,

        linestyle="--",

        linewidth=1.2,

        alpha=0.8,

        label="TRUST mean"

    )

    ax.errorbar(

        [1],

        [trust_y],

        yerr=[trust_err],

        marker="*",

        markersize=14,

        linewidth=1.2,

        capsize=4,

        label="TRUST",

        zorder=5

    )

# Axis formatting

xticks = [1] + sorted(baseline["cost"].unique().tolist())

ax.set_xticks(xticks)

ax.set_xticklabels([f"{int(x)}×" for x in xticks])

ax.set_xlabel("Relative reverse-time evaluations")

ax.set_ylabel("AUSE ↓")

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

# ax.legend(frameon=False, loc="best")

fig.tight_layout()

fig.savefig(OUT_PDF, bbox_inches="tight")

fig.savefig(OUT_PDF.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

print(f"Saved: {OUT_PDF}")

##############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr

root = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1_motion/uncertainty_eval/FM_007"

n = 8
FILES = {

    "TRUST": f"{root}/_TRUST.csv",

    "Post-hoc sampling": f"{root}/_PostHoc.csv",

}


UNC_COL = "u_top1_mean"

ERR_COL = "MAE"

OUT_PREFIX = "uncertainty_error_alignment"

def load_clean(path, unc_col=UNC_COL, err_col=ERR_COL):

    df = pd.read_csv(path)

    required = {unc_col, err_col}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"{path} missing columns: {missing}")

    df = df[[unc_col, err_col]].dropna()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

def isotonic_fit(x, y, n_grid=200):

    order = np.argsort(x)

    x_sorted = x[order]

    y_sorted = y[order]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    iso.fit(x_sorted, y_sorted)

    x_grid = np.linspace(x_sorted.min(), x_sorted.max(), n_grid)

    y_grid = iso.predict(x_grid)

    return x_grid, y_grid

summary_rows = []

data = {}

for method, path in FILES.items():

    df = load_clean(path)

    x = df[UNC_COL].to_numpy()

    y = df[ERR_COL].to_numpy()

    rho, p_s = spearmanr(x, y)

    r, p_p = pearsonr(x, y)

    summary_rows.append({

        "Method": method,

        "Spearman_rho": rho,

        "Spearman_p": p_s,

        "Pearson_r": r,

        "Pearson_p": p_p,

        "N": len(df),

    })

    data[method] = (x, y)

summary = pd.DataFrame(summary_rows)

summary.to_csv(f"{root}/{OUT_PREFIX}_correlations.csv", index=False)

print(summary)

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 11,

    "axes.labelsize": 12,

    "axes.titlesize": 12,

    "legend.fontsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)

for ax, (method, (x, y)) in zip(axes, data.items()):

    x_fit, y_fit = isotonic_fit(x, y)
    rho = summary.loc[summary["Method"] == method, "Spearman_rho"].iloc[0]

    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.18,
        label="samples",
        rasterized=True,
    )

    # ax.plot(
    #    x_fit,
    #     y_fit,
    #    linewidth=2.4,
    #    label="isotonic fit",
    # )

    ax.set_title(f"{method}\nSpearman $\\rho={rho:.3f}$")
    ax.set_xlabel("Mean predicted uncertainty")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

axes[0].set_ylabel("Mean absolute error")
axes[0].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(f"{root}/{OUT_PREFIX}_N_{n}.pdf", bbox_inches="tight")
fig.savefig(f"{root}/{OUT_PREFIX}_N_{n}.png", dpi=300, bbox_inches="tight")

print(f"Saved: {OUT_PREFIX}.pdf")
print(f"Saved: {OUT_PREFIX}.png")
print(f"Saved: {OUT_PREFIX}_correlations.csv")


########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CSV_PATH = "/Users/francescodifeola/Desktop/metrics_epoch_140_uncertainty_calibration_k_10.csv"
OUT_PREFIX = "calibration_curve"

df = pd.read_csv(CSV_PATH)
df = df[df["Type"] == "calibration"].copy()

# Weighted aggregation across samples
g = (
    df.groupby("Bin")
    .apply(lambda x: pd.Series({
        "Unc_mean": np.average(x["Unc_mean"], weights=x["Count"]),
        "Err_mean": np.average(x["Err_mean"], weights=x["Count"]),
        "Err_std": x["Err_mean"].std(),
        "Err_sem": x["Err_mean"].std() / np.sqrt(len(x)),
        "Count": x["Count"].sum()
    }))
    .reset_index()
)

# rho, p = spearmanr(g["Unc_mean"], g["Err_mean"])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(4.6, 3.3))

ax.errorbar(
    g["Unc_mean"],
    g["Err_mean"],
    yerr=g["Err_std"],
    marker="o",
    linewidth=2,
    capsize=3,
)

ax.set_xlabel("Mean predicted uncertainty")
ax.set_ylabel("Mean absolute error")
ax.set_xscale("log")
# ax.set_title(rf"Uncertainty-error alignment, Spearman $\rho={rho:.3f}$")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

fig.tight_layout()
fig.savefig(f"/Users/francescodifeola/Desktop/{OUT_PREFIX}.pdf", bbox_inches="tight")
fig.savefig(f"/Users/francescodifeola/Desktop/{OUT_PREFIX}.png", dpi=300, bbox_inches="tight")

g.to_csv(f"{OUT_PREFIX}_binned.csv", index=False)

####################
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

TRUST_CSV = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_TRUST.csv"
POSTHOC_CSV = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_PostHoc_N_8.csv"

OUT_PREFIX = "uncertainty_error_tail_bins_comparison"

OUT_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM"

def aggregate_tail_bins(csv_path):
    df = pd.read_csv(csv_path)
    g = (
        df.groupby(["bin", "p_low", "p_high"])
        .apply(lambda x: pd.Series({
            "mean_uncertainty": np.average(x["mean_uncertainty"], weights=x["count"]),
            "mean_error": np.average(x["mean_error"], weights=x["count"]),
            "std_error": x["mean_error"].std(),
            "sem_error": x["mean_error"].std() / np.sqrt(len(x)),
            "count": x["count"].sum(),
        }))
        .reset_index()
    )
    return g
g_trust = aggregate_tail_bins(TRUST_CSV)
g_posthoc = aggregate_tail_bins(POSTHOC_CSV)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.errorbar(
    g_trust["mean_uncertainty"],
    g_trust["mean_error"],
    yerr=g_trust["std_error"],
    marker="o",
    linewidth=2.0,
    capsize=3,
    label="TRUST",
)
ax.errorbar(
    g_posthoc["mean_uncertainty"],
    g_posthoc["mean_error"],
    yerr=g_posthoc["std_error"],
    marker="s",
    linewidth=2.0,
    capsize=3,
    label="Post-hoc sampling",
)

# Optional: annotate percentile bins only once, using TRUST positions
# for _, r in g_trust.iterrows():
#    ax.annotate(
#        f"{int(r['p_low'])}-{int(r['p_high'])}%",
#        (r["mean_uncertainty"], r["mean_error"]),
#        textcoords="offset points",
#        xytext=(4, 4),
#       fontsize=8,
#    )
ax.set_xlabel("Mean predicted uncertainty")
ax.set_ylabel("Mean absolute error")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/{OUT_PREFIX}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/{OUT_PREFIX}.png", dpi=300, bbox_inches="tight")
g_trust.to_csv(f"{OUT_DIR}/{OUT_PREFIX}_TRUST_aggregated.csv", index=False)
g_posthoc.to_csv(f"{OUT_DIR}/{OUT_PREFIX}_PostHoc_aggregated.csv", index=False)
print("TRUST")
print(g_trust)
print("\nPost-hoc sampling")
print(g_posthoc)

#######
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL = "LDM"  # DM, FM, LDM, LFM

BASE_DIR = f"/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/uncertainty_cal/{MODEL}"

TRUST_CSV = os.path.join(BASE_DIR, "_TRUST.csv")
POSTHOC_CSV = os.path.join(BASE_DIR, "_PostHoc_N_8.csv")

OUT_PREFIX = f"uncertainty_error_relative_{MODEL}"


def aggregate_tail_bins(csv_path):

    df = pd.read_csv(csv_path)

    g = (
        df.groupby(["bin", "p_low", "p_high"])
        .apply(lambda x: pd.Series({
            "mean_uncertainty": np.average(
                x["mean_uncertainty"],
                weights=x["count"]
            ),
            "mean_error": np.average(
                x["mean_error"],
                weights=x["count"]
            ),
            "std_error": x["mean_error"].std(),
            "sem_error": x["mean_error"].std() / np.sqrt(len(x)),
            "count": x["count"].sum(),
        }))
        .reset_index()
        .sort_values("bin")
    )

    return g


def add_relative_uncertainty(g):

    g = g.copy()

    g["relative_uncertainty"] = (
        g["mean_uncertainty"]
        /
        (g["mean_uncertainty"].max() + 1e-12)
    )

    return g


g_trust = aggregate_tail_bins(TRUST_CSV)
g_posthoc = aggregate_tail_bins(POSTHOC_CSV)

g_trust = add_relative_uncertainty(g_trust)
g_posthoc = add_relative_uncertainty(g_posthoc)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(4.8, 3.4))

# --------------------------------------------------
# TRUST
# --------------------------------------------------

ax.plot(
    g_trust["relative_uncertainty"],
    g_trust["mean_error"],
    marker="o",
    markersize=8,
    linewidth=2.5,
    color="orange",
    label="TRUST",
)

# --------------------------------------------------
# Post-hoc
# --------------------------------------------------

ax.plot(
    g_posthoc["relative_uncertainty"],
    g_posthoc["mean_error"],
    marker="s",
    markersize=8,
    linewidth=2.5,
    color="blue",
    label="Post-hoc sampling",
)

# --------------------------------------------------
# Formatting
# --------------------------------------------------

ax.set_xlabel("Relative predicted uncertainty")
ax.set_ylabel(r"$|y-\hat{y}|$")

ax.grid(
    False,
    linestyle="--",
    linewidth=0.6,
    alpha=0.5,
)

# ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}.pdf"),
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}.png"),
    dpi=300,
    bbox_inches="tight",
)

print("TRUST")
print(g_trust)

print("\nPost-hoc sampling")
print(g_posthoc)


######
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

TRUST_CSV = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_TRUST.csv"

POSTHOC_CSVS = {

    4: "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_PostHoc_N_4.csv",

    5: "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_PostHoc_N_5.csv",

    6: "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_PostHoc_N_6.csv",

    7: "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_PostHoc_N_7.csv",

    8: "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM/_PostHoc_N_8.csv",

}

OUT_PREFIX = "uncertainty_error_tail_bins_allN"

OUT_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/uncertainty_cal/FM"

def aggregate_tail_bins(csv_path):

    df = pd.read_csv(csv_path)

    g = (

        df.groupby(["bin", "p_low", "p_high"])

        .apply(lambda x: pd.Series({

            "mean_uncertainty": np.average(

                x["mean_uncertainty"],

                weights=x["count"]

            ),

            "mean_error": np.average(

                x["mean_error"],

                weights=x["count"]

            ),

            "std_error": x["mean_error"].std(),

            "sem_error": x["mean_error"].std() / np.sqrt(len(x)),

            "count": x["count"].sum(),

        }))

        .reset_index()

    )

    return g

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 11,

    "axes.labelsize": 12,

    "axes.titlesize": 12,

    "legend.fontsize": 9,

})

fig, ax = plt.subplots(figsize=(5.2, 3.8))

# ==================================================

# TRUST

# ==================================================

g_trust = aggregate_tail_bins(TRUST_CSV)

ax.errorbar(

    g_trust["mean_uncertainty"],

    g_trust["mean_error"],

    #yerr=g_trust["std_error"],

    marker="o",

    linewidth=3.0,

    capsize=3,

    label="TRUST",

)

g_trust.to_csv(

    f"{OUT_DIR}/{OUT_PREFIX}_TRUST_aggregated.csv",

    index=False

)

# ==================================================

# Post-hoc curves

# ==================================================

markers = ["s", "^", "v", "D", "P"]

for marker, (N, csv_path) in zip(markers, POSTHOC_CSVS.items()):

    g = aggregate_tail_bins(csv_path)

    ax.plot(

        g["mean_uncertainty"],

        g["mean_error"],

        marker=marker,

        linewidth=1.5,

        alpha=0.85,

        label=rf"Post-hoc ($N={N}$)",

    )

    g.to_csv(

        f"{OUT_DIR}/{OUT_PREFIX}_PostHoc_N_{N}_aggregated.csv",

        index=False

    )

# ==================================================

# Formatting

# ==================================================

ax.set_xlabel("Mean predicted uncertainty")

ax.set_ylabel("Mean absolute error")

ax.grid(

    True,

    linestyle="--",

    linewidth=0.6,

    alpha=0.5

)

ax.legend(

    frameon=False,

    ncol=2

)

fig.tight_layout()

fig.savefig(

    f"{OUT_DIR}/{OUT_PREFIX}.pdf",

    bbox_inches="tight"

)

fig.savefig(

    f"{OUT_DIR}/{OUT_PREFIX}.png",

    dpi=300,

    bbox_inches="tight"

)

print("Saved figure.")

############################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL = "LFM_02"  # DM, FM, LDM, LFM


col_ylims = {
    "DM_02":  (0.7, 2.5),
    "FM_02":  (0.7, 2.5),
    "LDM_02": (0.6, 3.6),
    "LFM_02": (0.6, 2.8),
}

BASE_DIR = f"/Users/francescodifeola/Desktop/omega/uncertainty/results/T1_motion/uncertainty_cal/{MODEL}"

TRUST_CSV = os.path.join(BASE_DIR, "_TRUST.csv")
POSTHOC_CSV = os.path.join(BASE_DIR, "_PostHoc_N_8.csv")

OUT_PREFIX = f"uncertainty_relative_error_{MODEL}"


def load_and_compute_relative_error(csv_path):
    df = pd.read_csv(csv_path)

    # df = df.iloc[:34987]

    required = {
        "sample",
        "bin",
        "p_low",
        "p_high",
        "mean_uncertainty",
        "mean_error",
        "count",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    # Reconstruct image-level MAE from bin-level statistics
    sample_mae = (
        df.groupby("sample")
        .apply(lambda x: np.sum(x["count"] * x["mean_error"]) / np.sum(x["count"]))
    )

    df["image_mae"] = df["sample"].map(sample_mae)

    # Relative error: bin error normalized by image-level MAE
    df["relative_error"] = df["mean_error"] / (df["image_mae"] + 1e-12)

    # Aggregate across samples
    g = (
        df.groupby(["bin", "p_low", "p_high"])
        .apply(lambda x: pd.Series({
            "mean_uncertainty": np.average(
                x["mean_uncertainty"],
                weights=x["count"]
            ),

            "relative_uncertainty": np.average(
                x["mean_uncertainty"],
                weights=x["count"]
            ),

            "mean_relative_error": np.average(
                x["relative_error"],
                weights=x["count"]
        ),

            "std_relative_error": x["relative_error"].std(),
            "sem_relative_error": x["relative_error"].std() / np.sqrt(len(x)),
            "count": x["count"].sum(),
        })

               )
        .reset_index()
        .sort_values("bin")
    )

    # Normalize uncertainty scale for visualization
    g["relative_uncertainty"] = (
        g["relative_uncertainty"] /
        (g["relative_uncertainty"].max() + 1e-12)
    )

    return df, g


df_trust, g_trust = load_and_compute_relative_error(TRUST_CSV)
df_posthoc, g_posthoc = load_and_compute_relative_error(POSTHOC_CSV)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
})

fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.plot(
    g_posthoc["relative_uncertainty"],
    g_posthoc["mean_relative_error"],
    marker="s",
    markersize=5,
    linewidth=2,
    # color="blue",
    label="Post-hoc sampling",
)
ax.plot(
    g_trust["relative_uncertainty"],
    g_trust["mean_relative_error"],
    marker="s",
    markersize=5,
    linewidth=2,
    # color="orange",
    label="TRUST",
)

ax.set_ylim(*col_ylims[MODEL])
ax.set_xlim(0, 1.05)
# ax.set_xticks([0.0, 0.5, 1.0])
ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    color="black",
    alpha=0.6,
)

ax.set_xlabel(r"$\tilde{\Sigma}_{\hat{\mathbf{y}},b}$")
ax.set_ylabel(r"$\mathrm{MAE}_{bin}/\mathrm{MAE}_{img}$")

ax.grid(
    False,
    linestyle="--",
    linewidth=0.6,
    alpha=0.5,
)

# ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}.pdf"),
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}.png"),
    dpi=300,
    bbox_inches="tight",
)

g_trust.to_csv(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}_TRUST_aggregated.csv"),
    index=False,
)

g_posthoc.to_csv(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}_PostHoc_N8_aggregated.csv"),
    index=False,
)

print("TRUST")
print(g_trust)

print("\nPost-hoc sampling")
print(g_posthoc)

######################################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODEL = "LFM"  # DM_02, FM_02, LDM_02, LFM_02

col_ylims = {
    "DM":  (0.7, 2.5),
    "FM":  (0.7, 2.5),
    "LDM": (0.6, 3.6),
    "LFM": (0.6, 2.8),
}

BASE_DIR = f"/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/uncertainty_cal/{MODEL}"

TRUST_CSV = os.path.join(BASE_DIR, "_TRUST.csv")
POSTHOC_CSV = os.path.join(BASE_DIR, "_PostHoc_N_8.csv")

OUT_PREFIX = f"uncertainty_relative_error_{MODEL}"

BIN_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    4: "v",
    5: "*",
}


def load_and_compute_relative_error(csv_path):
    df = pd.read_csv(csv_path)

    # df = df.iloc[:34987]

    required = {
        "sample",
        "bin",
        "p_low",
        "p_high",
        "mean_uncertainty",
        "mean_error",
        "count",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    sample_mae = (
        df.groupby("sample")
        .apply(lambda x: np.sum(x["count"] * x["mean_error"]) / np.sum(x["count"]))
    )

    df["image_mae"] = df["sample"].map(sample_mae)
    df["relative_error"] = df["mean_error"] / (df["image_mae"] + 1e-12)

    g = (
        df.groupby(["bin", "p_low", "p_high"])
        .apply(lambda x: pd.Series({
            "mean_uncertainty": np.average(
                x["mean_uncertainty"],
                weights=x["count"]
            ),
            "relative_uncertainty": np.average(
                x["mean_uncertainty"],
                weights=x["count"]
            ),
            "mean_relative_error": np.average(
                x["relative_error"],
                weights=x["count"]
            ),
            "std_relative_error": x["relative_error"].std(),
            "sem_relative_error": x["relative_error"].std() / np.sqrt(len(x)),
            "count": x["count"].sum(),
        }))
        .reset_index()
        .sort_values("bin")
    )

    g["relative_uncertainty"] = (
        g["relative_uncertainty"] /
        (g["relative_uncertainty"].max() + 1e-12)
    )

    return df, g


def plot_with_bin_markers(ax, g, color, label):
    ax.plot(
        g["relative_uncertainty"],
        g["mean_relative_error"],
        linewidth=2,
        color=color,
        label=label,
        zorder=2,
    )

    for _, row in g.iterrows():
        b = int(row["bin"])
        ax.scatter(
            row["relative_uncertainty"],
            row["mean_relative_error"],
            marker=BIN_MARKERS.get(b, "o"),
            s=45 if b != 5 else 80,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )


df_trust, g_trust = load_and_compute_relative_error(TRUST_CSV)
df_posthoc, g_posthoc = load_and_compute_relative_error(POSTHOC_CSV)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(4.8, 3.4))

plot_with_bin_markers(
    ax,
    g_posthoc,
    color="tab:blue",
    label="Post-hoc sampling",
)

plot_with_bin_markers(
    ax,
    g_trust,
    color="tab:orange",
    label="TRUST",
)

ax.set_ylim(*col_ylims[MODEL])
ax.set_xlim(0, 1.05)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    color="black",
    alpha=0.6,
)

ax.set_xlabel(r"$\tilde{\Sigma}_{\hat{\mathbf{y}},b}$")
ax.set_ylabel(r"$\frac{|\mathbf{y}-\hat{\mathbf{y}}|_{b}}{|\mathbf{y}-\hat{\mathbf{y}}|}$")

"""
method_legend = ax.legend(
    frameon=False,
    loc="upper left",
)

ax.add_artist(method_legend)
"""

bin_handles = []
for b, row in g_trust.iterrows():
    bin_handles.append(
        Line2D(
            [0],
            [0],
            marker=BIN_MARKERS.get(int(row["bin"]), "o"),
            color="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7 if int(row["bin"]) != 5 else 9,
            label=f"{int(row['p_low'])}-{int(row['p_high'])}%",
        )
    )

# ax.legend(
#     handles=bin_handles,
#     frameon=False,
#     loc="lower right",
#     title="Percentile bin",
#     fontsize=8,
#     title_fontsize=9,
# )

ax.grid(False)

fig.tight_layout()

fig.savefig(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}.pdf"),
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}.png"),
    dpi=300,
    bbox_inches="tight",
)

g_trust.to_csv(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}_TRUST_aggregated.csv"),
    index=False,
)

g_posthoc.to_csv(
    os.path.join(BASE_DIR, f"{OUT_PREFIX}_PostHoc_N8_aggregated.csv"),
    index=False,
)