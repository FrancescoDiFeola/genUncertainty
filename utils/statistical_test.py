import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/statistical_test/T1T2"
OUTPUT_FILE = "/Users/francescodifeola/Desktop/omega/uncertainty/results/statistical_test/T1T2/statistical_teststatistical_results.csv"

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

#################
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "/Users/francescodifeola/Desktop/omega/uncertainty/results/statistical_test/T1T2"
OUTPUT_FILE = os.path.join(ROOT_DIR, "statistical_results.csv")

METRICS = ["PSNR", "SSIM"]
ALPHA = 0.05

# ============================================================
# EFFECT SIZE (Rank Biserial Correlation)
# ============================================================

def rank_biserial(stat, n):
    max_stat = n * (n + 1) / 2
    return 1 - (2 * stat) / max_stat

# ============================================================
# PARSE FILES
# ============================================================

files = glob.glob(os.path.join(ROOT_DIR, "*.csv"))

models = {}

for f in files:
    name = os.path.basename(f).replace(".csv", "")
    parts = name.split("_")

    model = parts[0]
    condition = parts[1]  # ours or standard

    if model not in models:
        models[model] = {}

    models[model][condition] = f

# ============================================================
# RUN STATISTICAL TESTS
# ============================================================

rows = []
all_pvals = []

for model, cond in models.items():

    if "ours" not in cond or "standard" not in cond:
        continue

    print(f"Processing model: {model}")

    df_ours = pd.read_csv(cond["ours"]).sort_values("Sample")
    df_std = pd.read_csv(cond["standard"]).sort_values("Sample")

    for metric in METRICS:

        x = df_ours[metric].values
        y = df_std[metric].values

        if len(x) != len(y):
            raise ValueError(f"Mismatch in {model}-{metric}")

        # One-sided test: is Ours > Standard?
        stat, p_raw = wilcoxon(x, y, alternative="less")

        row = {
            "Model": model,
            "Metric": metric,
            "Mean_Ours": np.mean(x),
            "Mean_Standard": np.mean(y),
            "Delta": np.mean(x) - np.mean(y),
            "p_raw": p_raw,
            "Statistic": stat,
            "N": len(x),
        }

        rows.append(row)
        all_pvals.append(p_raw)

# ============================================================
# HOLM CORRECTION (ACROSS ALL TESTS)
# ============================================================

if len(all_pvals) > 0:

    reject, p_corr, _, _ = multipletests(
        all_pvals,
        alpha=ALPHA,
        method="holm"
    )

    for i, row in enumerate(rows):

        row["p_holm"] = p_corr[i]
        row["Significant"] = reject[i]

        if reject[i]:
            row["Decision"] = "Significant improvement"
        else:
            row["Decision"] = "No significant difference"

        row["Effect_size_r"] = rank_biserial(
            row["Statistic"],
            row["N"]
        )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nFinal Statistical Test Results:")
print(results_df)


import nibabel as nib
import numpy as np
import os


def denormalize_nifti(input_path, output_path, orig_min, orig_max):
    """
    Load a NIfTI file, check its range, denormalize it, and save it back.

    Args:
        input_path (str): path to normalized nifti
        output_path (str): path to save denormalized nifti
        orig_min (float): original minimum value before normalization
        orig_max (float): original maximum value before normalization
    """

    # Load nifti
    nii = nib.load(input_path)
    data = nii.get_fdata()

    # Print range
    print("Current range:")
    print("min:", np.min(data))
    print("max:", np.max(data))

    # Denormalize
    denorm = data * (orig_max - orig_min) + orig_min

    print("Denormalized range:")
    print("min:", np.min(denorm))
    print("max:", np.max(denorm))

    # Save nifti
    new_nii = nib.Nifti1Image(denorm, affine=nii.affine, header=nii.header)
    nib.save(new_nii, output_path)

    print(f"Saved to {output_path}")




input_nifti = "/Users/francescodifeola/Desktop/omega/DEMO_DigitalTwin/patients_analysis/tasks_UI/MRI_to_CT_pelvis/patients/1PA001/p.nii.gz"
output_nifti = "/Users/francescodifeola/Desktop/omega/DEMO_DigitalTwin/patients_analysis/tasks_UI/MRI_to_CT_pelvis/patients/1PA001/p_deno.nii.gz"

# Example: original intensity range
orig_min = -1024
orig_max = 1478

denormalize_nifti(input_nifti, output_nifti, orig_min, orig_max)