import matplotlib
matplotlib.use("TkAgg")  # important on macOS before importing pyplot
from src.inference.utils import compute_aurg, compute_ause
import pandas as pd
import matplotlib.pyplot as plt

# VIOLIN PLOTS

df_ddpm = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/RF/RF_T1T2_metrics_epoch_300.csv")  # "/Users/francescodifeola/Desktop/omega/uncertainty/ddpm_b16_T1T2/300/ddpm_b16_T1T2_metrics.csv"
df_alea = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/RF_aleatoric/RF_T1T2_aleatoric_metrics_epoch_900.csv") # "/Users/francescodifeola/Desktop/omega/uncertainty/aleatoric_uncertainty_b16_T1T2/300/aleatoric_uncertainty_b16_T1T2_metrics.csv"
df_twof = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/RF_aleatoric_two_forward/RF_T1T2_aleatoric_two_forward_metrics_iterative_refinement_without_twoforward_epoch_900.csv")  # "/Users/francescodifeola/Desktop/omega/uncertainty/aleatoric_uncertainty_cross_attention_T1T2/300/aleatoric_uncertainty_cross_attention_T1T2_metrics.csv"

df_ddpm["Model"] = "DDPM"
df_alea["Model"] = "Aleatoric"
df_twof["Model"] = "TwoForward"

df_all = pd.concat([df_ddpm, df_alea, df_twof], ignore_index=True)

metrics = ["MSE", "PSNR", "SSIM"]
models = ["DDPM", "Aleatoric", "TwoForward"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric in zip(axes, metrics):
    data = [df_all.loc[df_all["Model"] == m, metric].dropna().astype(float)
            for m in models]

    ax.violinplot(data, showmeans=False, showmedians=True)
    ax.set_title(metric)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models)
    ax.grid(True)

plt.tight_layout()
print("About to show plot")
plt.show()


###########################
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/DDPM/baseline/metrics_epoch_300.csv")

mse = (df["MSE"].mean(), df["MSE"].std())
psnr = (df["PSNR"].mean(), df["PSNR"].std())
ssim = (df["SSIM"].mean(), df["SSIM"].std())
# pearson = (df["Pearson_u_norm"].mean(), df["Pearson_u_norm"].std())
# spearman = (df["Spearman_u_norm"].mean(), df["Spearman_u_norm"].std())
# auc_top15 = (df["AUROC_top15_u_norm"].mean(), df["AUROC_top15_u_norm"].std())
# auc_top10 = (df["AUROC_top10_u_norm"].mean(), df["AUROC_top10_u_norm"].std())
# auc_top5 = (df["AUROC_top5_u_norm"].mean(), df["AUROC_top5_u_norm"].std())


# print(mse, psnr, ssim) #  pearson, spearman, pearson_norm, spearman_norm)
print(
    f"SSIM: mean={ssim[0]:.3f}, std={ssim[1]:.3f} | "
    f"PSNR: mean={psnr[0]:.3f}, std={psnr[1]:.3f} | "
    # f"Pearson: mean={pearson[0]:.3f}, std={pearson[1]:.3f} | "
    # f"Spearman: mean={spearman[0]:.3f}, std={spearman[1]:.3f} |  "
    # f"AUC_top15: mean={auc_top15[0]:.3f}, std={auc_top15[1]:.3f} | "
    # f"AUC_top10: mean={auc_top10[0]:.3f}, std={auc_top10[1]:.3f} | "
    # f"AUC_top5: mean={auc_top5[0]:.3f}, std={auc_top5[1]:.3f} | "
)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# 1. Load CSV
# =========================================
df = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/RF/T1T2_Brats/metrics_epoch_50.csv")

# =========================================
# 2. LaTeX-style Configuration
# =========================================
plt.rcParams.update({
    "text.usetex": False,            # Set to True only if LaTeX is installed
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.5,
})

# =========================================
# 3. Create histogram function
# =========================================
def plot_hist(metric, bins=30):
    plt.figure(figsize=(6, 4))
    plt.hist(df[metric], bins=bins, color="tab:orange", edgecolor="black", alpha=0.85)

    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"Histogram of {metric}")

    # Grid and layout
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# =========================================
# 4. Generate the two histograms
# =========================================
plot_hist("Pearson")
plot_hist("Spearman")



####################################

import pandas as pd

def summarize_metrics(csv_path):
    df = pd.read_csv(csv_path)

    summary = {
        "n_samples": len(df),
        "MSE": {
            "mean": df["MSE"].mean(),
            "std": df["MSE"].std(),
            "median": df["MSE"].median(),
            "min": df["MSE"].min(),
            "max": df["MSE"].max(),
        },
        "PSNR": {
            "mean": df["PSNR"].mean(),
            "std": df["PSNR"].std(),
            "median": df["PSNR"].median(),
            "min": df["PSNR"].min(),
            "max": df["PSNR"].max(),
        },
        "SSIM": {
            "mean": df["SSIM"].mean(),
            "std": df["SSIM"].std(),
            "median": df["SSIM"].median(),
            "min": df["SSIM"].min(),
            "max": df["SSIM"].max(),
        },
    }

    return summary


def print_summary(summary, epoch=None):
    title = f"Summary (Epoch {epoch})" if epoch is not None else "Summary"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(f"Samples: {summary['n_samples']}\n")

    for metric in ["MSE", "PSNR", "SSIM"]:
        m = summary[metric]
        print(
            f"{metric}: "
            f"mean ± std = {m['mean']:.4f} ± {m['std']:.4f}, "
            f"median = {m['median']:.4f}, "
            f"range = [{m['min']:.4f}, {m['max']:.4f}]"
        )



root = "/Users/francescodifeola/Desktop/omega/uncertainty/results/RF/T1T2_Brats"
csv_path = f"{root}/metrics_epoch_100.csv"
epoch = 100

summary = summarize_metrics(csv_path)
print_summary(summary, epoch)


###############################

import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/MRCT/LDM"   # <-- change this
OUTPUT_CSV = "/Users/francescodifeola/Desktop/omega/uncertainty/results/MRCT/LDM/summary_metrics.csv"

METRICS = [
    "MSE",
    "PSNR",
    "SSIM",
    "Pearson_u_norm",
    "Spearman_u_norm",
    "AUROC_top15_u_norm",
    "AUROC_top10_u_norm",
    "AUROC_top5_u_norm",
]

# ============================================================
# HELPERS
# ============================================================

def mean_std_str(x):
    return f"{x.mean():.3f} ± {x.std():.3f}"

def parse_epoch_and_split(filename):
    """
    metrics_epoch_100.csv           -> epoch=100, split=test
    metrics_epoch_100_train.csv     -> epoch=100, split=train
    """
    name = os.path.basename(filename)
    split = "train" if "train" in name else "test"
    epoch = int(name.split("epoch_")[1].split("_")[0].split(".")[0])
    return epoch, split

# ============================================================
# MAIN
# ============================================================

rows = []

for experiment in sorted(os.listdir(ROOT_DIR)):
    exp_path = os.path.join(ROOT_DIR, experiment)
    if not os.path.isdir(exp_path):
        continue

    csv_files = glob.glob(os.path.join(exp_path, "metrics_epoch_*.csv"))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        epoch, split = parse_epoch_and_split(csv_path)

        row = {
            "Experiment": experiment,
            "Epoch": epoch,
            "Split": split,
        }

        for m in METRICS:
            if m in df.columns:
                row[m] = mean_std_str(df[m])

        rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(
    ["Experiment", "Epoch", "Split"]
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(OUTPUT_CSV, index=False)

print("\n===== SUMMARY TABLE =====\n")
print(summary_df)
print(f"\nSaved to {OUTPUT_CSV}")

df = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/T1T2/LFM/self_refinement_k10/metrics_epoch_50_K10_ablation_only_small_perturb.csv")


###############################
import pandas as pd

df = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/DDPM/baseline/self_refining/sparsification_epoch_300.csv")


# df_m = df[df['Method'] == method]
grouped = df.groupby('Fraction').mean()

fractions = grouped.index.values
mean_curve = grouped['Error'].values
mean_random = grouped['RandomError'].values
mean_oracle = grouped['OracleError'].values

ause = compute_ause(fractions, mean_curve)
ause_r = compute_ause(fractions, mean_random)
ause_o = compute_ause(fractions, mean_oracle)
aurg_o = compute_aurg(fractions, mean_oracle, mean_random)
aurg = compute_aurg(fractions, mean_curve, mean_random)

print("AUSE_our:", ause, "AUSE_oracle:", ause_o, "AUSE_random:", ause_r, "AURG_our:", aurg, "AURG_oracle:", aurg_o)

plt.figure()

plt.plot(fractions, mean_curve, label="Our")
# plt.plot(fractions, mean_curve_mc, label="MC-Sampling")
plt.plot(fractions, mean_random, linestyle='--', label="Random")
plt.plot(fractions, mean_oracle, linestyle=':', label="Oracle")

plt.xlabel("Fraction of removed pixels")
plt.ylabel("Remaining L1 Error")
plt.legend()
plt.grid(True)
plt.show()