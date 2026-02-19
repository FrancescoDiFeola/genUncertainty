import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/statistical_test"
OUTPUT_FILE = "/Users/francescodifeola/Desktop/omega/uncertainty/results/statistical_teststatistical_results.csv"

METRICS = ["PSNR", "SSIM"]
ALPHA = 0.05

# ============================================================
# HELPERS
# ============================================================

def rank_biserial(stat, n):
    max_stat = n * (n + 1) / 2
    return 1 - (2 * stat) / max_stat


def parse_filename(path):
    name = os.path.basename(path).replace(".csv", "")
    parts = name.split("_")
    model = parts[0]
    condition = parts[1]
    return model, condition

# ============================================================
# MAIN
# ============================================================

all_results = []

for task in sorted(os.listdir(ROOT_DIR)):

    task_path = os.path.join(ROOT_DIR, task)
    if not os.path.isdir(task_path):
        continue

    print(f"\nProcessing task: {task}")

    files = glob.glob(os.path.join(task_path, "*.csv"))

    models = {}

    for f in files:
        model, condition = parse_filename(f)
        if model not in models:
            models[model] = {}
        models[model][condition] = f

    task_rows = []
    task_pvals = []

    # --------------------------------------------------------
    # Run both directional tests
    # --------------------------------------------------------

    for model, cond in models.items():

        if "standard" not in cond or "ours" not in cond:
            continue

        df_std = pd.read_csv(cond["standard"]).sort_values("Sample")
        df_ours = pd.read_csv(cond["ours"]).sort_values("Sample")

        for metric in METRICS:

            x = df_ours[metric].values
            y = df_std[metric].values

            if len(x) != len(y):
                raise ValueError(f"Mismatch in {task}-{model}-{metric}")

            stat_greater, p_greater = wilcoxon(x, y, alternative="greater")
            stat_less, p_less = wilcoxon(x, y, alternative="less")

            row = {
                "Task": task,
                "Model": model,
                "Metric": metric,
                "Mean_Ours": np.mean(x),
                "Mean_Standard": np.mean(y),
                "p_greater_raw": p_greater,
                "p_less_raw": p_less,
                "Statistic_greater": stat_greater,
                "Statistic_less": stat_less,
                "N": len(x),
            }

            task_rows.append(row)

            # store both p-values for correction
            task_pvals.append(p_greater)
            task_pvals.append(p_less)

    # --------------------------------------------------------
    # Holm correction within task (all directional tests)
    # --------------------------------------------------------

    if len(task_pvals) > 0:

        reject, p_corr, _, _ = multipletests(
            task_pvals,
            alpha=ALPHA,
            method="holm"
        )

        # Assign corrected p-values back
        idx = 0
        for row in task_rows:

            row["p_greater_holm"] = p_corr[idx]
            row["greater_significant"] = reject[idx]
            idx += 1

            row["p_less_holm"] = p_corr[idx]
            row["less_significant"] = reject[idx]
            idx += 1

            # ------------------------------------------------
            # Final decision
            # ------------------------------------------------

            if row["greater_significant"]:
                row["Decision"] = "Better"
            elif row["less_significant"]:
                row["Decision"] = "Worse"
            else:
                row["Decision"] = "No difference"

            # Effect size (use greater direction)
            row["Effect_size_r"] = rank_biserial(
                row["Statistic_greater"], row["N"]
            )

    all_results.extend(task_rows)

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(all_results)

if not results_df.empty:
    results_df = results_df.sort_values(
        ["Task", "Model", "Metric"]
    ).reset_index(drop=True)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n===== DONE =====")
print(results_df)
print(f"\nSaved to {OUTPUT_FILE}")

# ============================================================
# HELPERS
# ============================================================

def rank_biserial(stat, n):
    max_stat = n * (n + 1) / 2
    return 1 - (2 * stat) / max_stat


def parse_filename(path):
    name = os.path.basename(path).replace(".csv", "")
    parts = name.split("_")
    model = parts[0]
    condition = parts[1]
    return model, condition


# ============================================================
# MAIN
# ============================================================

all_results = []

for task in sorted(os.listdir(ROOT_DIR)):

    task_path = os.path.join(ROOT_DIR, task)
    if not os.path.isdir(task_path):
        continue

    print(f"\nProcessing task: {task}")

    files = glob.glob(os.path.join(task_path, "*.csv"))

    models = {}

    for f in files:
        model, condition = parse_filename(f)
        if model not in models:
            models[model] = {}
        models[model][condition] = f

    task_rows = []
    task_pvals = []

    # --------------------------------------------------------
    # Run both directional tests
    # --------------------------------------------------------

    for model, cond in models.items():

        if "standard" not in cond or "ours" not in cond:
            continue

        df_std = pd.read_csv(cond["standard"]).sort_values("Sample")
        df_ours = pd.read_csv(cond["ours"]).sort_values("Sample")

        for metric in METRICS:

            x = df_ours[metric].values
            y = df_std[metric].values

            if len(x) != len(y):
                raise ValueError(f"Mismatch in {task}-{model}-{metric}")

            stat_greater, p_greater = wilcoxon(x, y, alternative="greater")
            stat_less, p_less = wilcoxon(x, y, alternative="less")

            row = {
                "Task": task,
                "Model": model,
                "Metric": metric,
                "Mean_Ours": np.mean(x),
                "Mean_Standard": np.mean(y),
                "p_greater_raw": p_greater,
                "p_less_raw": p_less,
                "Statistic_greater": stat_greater,
                "Statistic_less": stat_less,
                "N": len(x),
            }

            task_rows.append(row)

            # store both p-values for correction
            task_pvals.append(p_greater)
            task_pvals.append(p_less)

    # --------------------------------------------------------
    # Holm correction within task (all directional tests)
    # --------------------------------------------------------

    if len(task_pvals) > 0:

        reject, p_corr, _, _ = multipletests(
            task_pvals,
            alpha=ALPHA,
            method="holm"
        )

        # Assign corrected p-values back
        idx = 0
        for row in task_rows:

            row["p_greater_holm"] = p_corr[idx]
            row["greater_significant"] = reject[idx]
            idx += 1

            row["p_less_holm"] = p_corr[idx]
            row["less_significant"] = reject[idx]
            idx += 1

            # ------------------------------------------------
            # Final decision
            # ------------------------------------------------

            if row["greater_significant"]:
                row["Decision"] = "Better"
            elif row["less_significant"]:
                row["Decision"] = "Worse"
            else:
                row["Decision"] = "No difference"

            # Effect size (use greater direction)
            row["Effect_size_r"] = rank_biserial(
                row["Statistic_greater"], row["N"]
            )

    all_results.extend(task_rows)

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(all_results)

if not results_df.empty:
    results_df = results_df.sort_values(
        ["Task", "Model", "Metric"]
    ).reset_index(drop=True)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n===== DONE =====")
print(results_df)
print(f"\nSaved to {OUTPUT_FILE}")



# ============================================================
# HELPERS
# ============================================================

def rank_biserial(stat, n):
    max_stat = n * (n + 1) / 2
    return 1 - (2 * stat) / max_stat


def parse_filename(path):
    name = os.path.basename(path).replace(".csv", "")
    parts = name.split("_")
    model = parts[0]
    condition = parts[1]  # standard or ours
    return model, condition


# ============================================================
# MAIN
# ============================================================

all_results = []

for task in sorted(os.listdir(ROOT_DIR)):

    task_path = os.path.join(ROOT_DIR, task)
    if not os.path.isdir(task_path):
        continue

    print(f"\nProcessing task: {task}")

    files = glob.glob(os.path.join(task_path, "*.csv"))

    models = {}

    for f in files:
        model, condition = parse_filename(f)
        if model not in models:
            models[model] = {}
        models[model][condition] = f

    task_pvals = []
    task_rows = []

    # --------------------------------------------------------
    # Compute raw p-values
    # --------------------------------------------------------

    for model, cond in models.items():

        if "standard" not in cond or "ours" not in cond:
            continue

        df_std = pd.read_csv(cond["standard"])
        df_ours = pd.read_csv(cond["ours"])

        # ensure alignment by Sample
        df_std = df_std.sort_values("Sample")
        df_ours = df_ours.sort_values("Sample")

        for metric in METRICS:

            x = df_ours[metric].values
            y = df_std[metric].values

            if len(x) != len(y):
                raise ValueError(f"Mismatch in {task}-{model}-{metric}")

            stat, p = wilcoxon(x, y, alternative="greater")

            task_pvals.append(p)

            task_rows.append({
                "Task": task,
                "Model": model,
                "Metric": metric,
                "Mean_Ours": np.mean(x),
                "Mean_Standard": np.mean(y),
                "Raw_p": p,
                "Statistic": stat,
                "N": len(x)
            })

    # --------------------------------------------------------
    # Holm correction per task
    # --------------------------------------------------------

    if len(task_pvals) > 0:

        reject, p_corr, _, _ = multipletests(
            task_pvals,
            alpha=ALPHA,
            method="holm"
        )

        for i in range(len(task_rows)):
            task_rows[i]["p_Holm"] = p_corr[i]
            task_rows[i]["Significant"] = reject[i]

            stat = task_rows[i]["Statistic"]
            n = task_rows[i]["N"]
            task_rows[i]["Effect_size_r"] = rank_biserial(stat, n)

    all_results.extend(task_rows)

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(
    ["Task", "Model", "Metric"]
).reset_index(drop=True)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n===== DONE =====")
print(results_df)
print(f"\nSaved to {OUTPUT_FILE}")

