import matplotlib
matplotlib.use("TkAgg")  # important on macOS before importing pyplot

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

df = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/DDPM_aleatoric_two_forward/two_forward_variance_normalized_T1T2_metrics_iterative_refinement_without_twoforward_epoch_300.csv")

mse = (df["MSE"].mean(), df["MSE"].std())
psnr = (df["PSNR"].mean(), df["PSNR"].std())
ssim = (df["SSIM"].mean(), df["SSIM"].std())

print(mse, psnr, ssim)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# 1. Load CSV
# =========================================
df = pd.read_csv("/Users/francescodifeola/Desktop/omega/uncertainty/DDPM_aleatoric_two_forward/two_forward_variance_normalized_denoising_metrics_iterative_refinement_without_twoforward_epoch_300.csv")

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
