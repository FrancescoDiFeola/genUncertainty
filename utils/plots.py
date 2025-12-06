import pandas as pd
import matplotlib.pyplot as plt

# VIOLIN PLOTS

# Load all three model CSVs
df_ddpm = pd.read_csv("/mnt/data/ddpm_b16_T1T2_metrics.csv")
df_alea = pd.read_csv("/mnt/data/aleatoric_uncertainty_b16_T1T2_metrics_300.csv")
df_twof = pd.read_csv("/mnt/data/two_forward_variance_normalized_T1T2_metrics_iterative_refinement_epoch_300.csv")

# Assign model column
df_ddpm["Model"] = "DDPM"
df_alea["Model"] = "Aleatoric"
df_twof["Model"] = "TwoForward"

# Concatenate
df_all = pd.concat([df_ddpm, df_alea, df_twof], ignore_index=True)

metrics = ["MSE", "PSNR", "SSIM"]

# Create violin plots: one per metric showing the 3 models
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric in zip(axes, metrics):
    data = [df_all[df_all["Model"] == m][metric] for m in ["DDPM", "Aleatoric", "TwoForward"]]

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True
    )

    ax.set_title(metric)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["DDPM", "Aleatoric", "TwoForward"])
    ax.grid(True)

plt.tight_layout()
plt.show()