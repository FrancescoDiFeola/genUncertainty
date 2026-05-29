import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

root = "/Users/francescodifeola/Desktop/omega/uncertainty/results/denoising/noise_levels/ddpm"  # contains baseline/ and self_refining/

models = {
    "Baseline": "baseline",
    "SelfRefining": "self_refining"
}

noise_types = ["gaussian", "uniform", "impulse"]
levels = [0, 1, 2, 3]

eta_values = {
    "gaussian": [0, 0.10, 0.20, 0.30],
    "uniform":  [0, 0.20, 0.40, 0.60],
    "impulse":  [0, 0.15, 0.30, 0.45]
}

# --------------------------------------------------
# Load metrics for a given model + noise
# --------------------------------------------------

def load_model_noise(model_folder, noise_type):

    results = []

    for lvl in levels:
        filename = f"metrics_epoch_300_{noise_type}_level_{lvl}.csv"
        path = os.path.join(root, model_folder, filename)

        df = pd.read_csv(path)

        results.append({
            "level": lvl,
            "eta": eta_values[noise_type][lvl-1],
            "PSNR_mean": df["PSNR"].mean(),
            "PSNR_std": df["PSNR"].std(ddof=1),
            "SSIM_mean": df["SSIM"].mean(),
            "SSIM_std": df["SSIM"].std(ddof=1),
            "raw": df  # keep raw for stats
        })

    return results


# --------------------------------------------------
# AUC computation
# --------------------------------------------------

def compute_auc(x, y):
    return np.trapz(y, x)


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

summary_rows = []

for noise in noise_types:

    print(f"\nProcessing: {noise}")

    model_data = {}

    for model_name, folder in models.items():
        model_data[model_name] = load_model_noise(folder, noise)

    # -------------------------
    # Extract curves
    # -------------------------

    x = eta_values[noise]

    mse_curves = {}
    ssim_curves = {}

    for model_name in models.keys():
        mse_curves[model_name] = [
            d["PSNR_mean"] for d in model_data[model_name]
        ]
        ssim_curves[model_name] = [
            d["SSIM_mean"] for d in model_data[model_name]
        ]

    # -------------------------
    # Compute AUC
    # -------------------------

    mse_auc = {
        model: compute_auc(x, mse_curves[model])
        for model in models.keys()
    }

    ssim_auc = {
        model: compute_auc(x, ssim_curves[model])
        for model in models.keys()
    }

    # -------------------------
    # Statistical test (paired)
    # -------------------------

    p_values_mse = []
    p_values_ssim = []

    for lvl_idx in range(len(levels)):

        df_base = model_data["Baseline"][lvl_idx]["raw"]
        df_self = model_data["SelfRefining"][lvl_idx]["raw"]

        t_mse = ttest_rel(df_base["PSNR"], df_self["PSNR"])
        t_ssim = ttest_rel(df_base["SSIM"], df_self["SSIM"])

        p_values_mse.append(t_mse.pvalue)
        p_values_ssim.append(t_ssim.pvalue)

    # -------------------------
    # Save summary
    # -------------------------

    summary_rows.append({
        "Noise": noise,
        "Baseline_PSNR_AUC": mse_auc["Baseline"],
        "SelfRefining_PSNR_AUC": mse_auc["SelfRefining"],
        "Baseline_SSIM_AUC": ssim_auc["Baseline"],
        "SelfRefining_SSIM_AUC": ssim_auc["SelfRefining"],
        "Mean_p_PSNR": np.mean(p_values_mse),
        "Mean_p_SSIM": np.mean(p_values_ssim)
    })

    # -------------------------
    # Plot MSE curve
    # -------------------------

    plt.figure(figsize=(6,5))
    for model_name in models.keys():
        plt.plot(x, mse_curves[model_name], marker="o", label=model_name)

    plt.xlabel("Noise Magnitude")
    plt.ylabel("Mean PSNR")
    plt.title(f"PSNR vs Noise ({noise})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{root}/{noise}_PSNR_comparison.pdf", dpi=300)
    plt.close()

    # -------------------------
    # Plot SSIM curve
    # -------------------------

    plt.figure(figsize=(6,5))
    for model_name in models.keys():
        plt.plot(x, ssim_curves[model_name], marker="o", label=model_name)

    plt.xlabel("Noise Magnitude")
    plt.ylabel("Mean SSIM")
    plt.title(f"SSIM vs Noise ({noise})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{root}/{noise}_SSIM_comparison.pdf", dpi=300)
    plt.close()


# --------------------------------------------------
# Final Summary Table
# --------------------------------------------------

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{root}/robustness_summary_comparison.csv", index=False)

print("\nFinal Robustness Summary:")
print(summary_df)