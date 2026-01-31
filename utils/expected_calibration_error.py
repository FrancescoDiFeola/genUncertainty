import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_ece_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Total number of pixels
    N = df["Count"].sum()

    # Per-bin absolute calibration gap
    df["calibration_gap"] = np.abs(df["Err_mean"] - df["Unc_mean"])

    # Weighted ECE
    ece = (df["Count"] / N * df["calibration_gap"]).sum()

    return ece



def plot_reliability_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(4, 4))
    plt.scatter(df["Unc_mean"], df["Err_mean"], marker="o", label="Model")

    # Perfect calibration line
    # max_val = max(df["Unc_mean"].max(), df["Err_mean"].max())
    # plt.plot([0, max_val], [0, max_val], "--", color="gray", label="Ideal")

    plt.xlabel("Predicted uncertainty")
    plt.ylabel("Empirical error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


ece = compute_ece_from_csv("/Users/francescodifeola/Desktop/aleatoric_uncertainty_b16_T1T2_metrics_epoch_300_uncertainty_calibration.csv")
print(ece)
plot_reliability_from_csv("/Users/francescodifeola/Desktop/aleatoric_uncertainty_b16_T1T2_metrics_epoch_300_uncertainty_calibration.csv")