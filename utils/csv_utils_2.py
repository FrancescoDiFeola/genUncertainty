import os
import csv

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
DATASET_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task1/pelvis/test"   # <-- change this
OUTPUT_CSV = "mr_ct_dataset_test.csv"

MR_PREFIXES = ("mr_", "MR_")
CT_PREFIXES = ("ct_", "CT_")

# ------------------------------------------------------
# Scan dataset and build rows
# ------------------------------------------------------
rows = []

for subject_id in sorted(os.listdir(DATASET_ROOT)):
    subject_path = os.path.join(DATASET_ROOT, subject_id)

    if not os.path.isdir(subject_path):
        continue

    for fname in sorted(os.listdir(subject_path)):
        fpath = os.path.join(subject_path, fname)

        if not os.path.isfile(fpath):
            continue

        if fname.endswith(".npy"):
            if fname.startswith(MR_PREFIXES):
                modality = "MR"
            elif fname.startswith(CT_PREFIXES):
                modality = "CT"
            else:
                continue  # skip unrelated files

            img_name = f"{subject_id}_{os.path.splitext(fname)[0]}"

            rows.append({
                "img_name": img_name,
                "img_path": fpath,
                "modality": modality
            })

# ------------------------------------------------------
# Write CSV
# ------------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["img_name", "img_path", "modality"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CSV with {len(rows)} entries to: {OUTPUT_CSV}")