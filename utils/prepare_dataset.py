import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

INPUT_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/SynthRAD2023_dataset/Task2//Task2/pelvis"
OUTPUT_ROOT = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/diffusion/Data/SynthRad2023/Task2/pelvis"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    # voxel spacing (x, y, z) in mm
    spacing = img.header.get_zooms()[:3]
    return data, spacing

subjects = sorted(os.listdir(INPUT_ROOT))

for subject in tqdm(subjects):
    if subject in {"persistent", "overview", "meta"}:
        continue
    subject_dir = os.path.join(INPUT_ROOT, subject)
    if not os.path.isdir(subject_dir):
        continue

    mr_path = os.path.join(subject_dir, "cbct.nii.gz")
    ct_path = os.path.join(subject_dir, "ct.nii.gz")

    if not (os.path.exists(mr_path) and os.path.exists(ct_path)):
        continue

    mr, spacing_mr = load_nii(mr_path)
    ct, spacing_ct = load_nii(ct_path)

    print(f"Subject: {subject}, MR: {spacing_mr}, CT: {spacing_ct}")
    assert mr.shape == ct.shape, f"Shape mismatch for {subject}"

    # out_subject_dir = os.path.join(OUTPUT_ROOT, subject)
    # os.makedirs(out_subject_dir, exist_ok=True)

    # for d in range(mr.shape[2]):
    #    np.save(os.path.join(out_subject_dir, f"cbct_{d:03d}.npy"), mr[:, :, d])
    #    np.save(os.path.join(out_subject_dir, f"ct_{d:03d}.npy"), ct[:, :, d])


"""
import numpy as np
import matplotlib.pyplot as plt

# Paths to your .npy files
mr_path = "/Users/francescodifeola/Desktop/cbct_048.npy"
ct_path = "/Users/francescodifeola/Desktop/ct_048.npy"

# Load arrays
mr = np.load(mr_path)
ct = np.load(ct_path)

# Optional: squeeze singleton dimensions
mr = np.squeeze(mr)
ct = np.squeeze(ct)

# Visualize side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(mr.T, cmap="gray")
axes[0].set_title("MR")
axes[0].axis("off")

axes[1].imshow(ct.T, cmap="gray")
axes[1].set_title("CT")
axes[1].axis("off")

plt.tight_layout()
plt.show()
"""