
import os

def build_paired_list(root_dir):
    """
    Returns a list of dicts:
    [
      {"mr": ".../mr_070.npy", "ct": ".../ct_070.npy"},
      ...
    ]
    """
    data = []

    for subject in sorted(os.listdir(root_dir)):
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        mr_files = sorted(f for f in os.listdir(subject_dir) if f.startswith("mr_"))

        for mr_file in mr_files:
            idx = mr_file.split("_")[1].split(".")[0]
            ct_file = f"ct_{idx}.npy"

            mr_path = os.path.join(subject_dir, mr_file)
            ct_path = os.path.join(subject_dir, ct_file)

            if os.path.exists(ct_path):
                data.append({
                    "mr": mr_path,
                    "ct": ct_path,
                })

    return data


def build_single_image_list(root_dir):
    """
    Returns:
    [
      {"img": ".../mr_070.npy", "modality": "mr"},
      {"img": ".../ct_070.npy", "modality": "ct"},
      ...
    ]
    """
    data = []

    for subject in sorted(os.listdir(root_dir)):
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        for fname in sorted(os.listdir(subject_dir)):
            if fname.startswith("mr_") and fname.endswith(".npy"):
                data.append({
                    "img": os.path.join(subject_dir, fname),
                    "modality": "mr",
                })

            elif fname.startswith("ct_") and fname.endswith(".npy"):
                data.append({
                    "img": os.path.join(subject_dir, fname),
                    "modality": "ct",
                })

    return data


