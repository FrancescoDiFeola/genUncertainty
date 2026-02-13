import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import pickle
from sklearn.model_selection import train_test_split



def is_informative_slice(path, threshold=0.05):
    """Return True if more than `threshold` fraction of pixels are non-zero."""
    arr = np.load(path)
    nonzero_fraction = np.count_nonzero(arr) / arr.size
    return nonzero_fraction >= threshold


def index_slices_by_subject(data_path, modalities=["t1n", "t1c", "t2w", "t2f"], min_idx=0, max_idx=999):
    valid_modalities = {"t1n", "t1c", "t2w", "t2f", "CT", "PET"}
    selected_modalities = [m for m in modalities if m in valid_modalities]
    if len(selected_modalities) < 2:
        raise ValueError("At least two valid modalities must be selected.")

    cache_prefix = f"{selected_modalities[0]}_to_{selected_modalities[1]}"
    samples_cache_path = os.path.join(data_path, f"{cache_prefix}_samples.json")
    subject_dict_cache_path = os.path.join(data_path, f"{cache_prefix}_subject_dict.pkl")

    if os.path.exists(samples_cache_path) and os.path.exists(subject_dict_cache_path):
        print(f"Caricamento cache da {samples_cache_path} e {subject_dict_cache_path}")
        with open(samples_cache_path, "r") as f:
            samples = json.load(f)
        with open(subject_dict_cache_path, "rb") as f:
            subject_dict = pickle.load(f)
        return samples, subject_dict

    main_modality = selected_modalities[0]
    subject_dict = {}

    for patient in tqdm(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient)
        if not os.path.isdir(patient_path):
            continue

        # Primo controllo: struttura piatta
        flat_main_folder = os.path.join(patient_path, main_modality)
        if os.path.isdir(flat_main_folder):
            series_list = [("", patient_path)]
        else:
            # Altrimenti cerchiamo sottocartelle (struttura annidata)
            series_list = [(series, os.path.join(patient_path, series)) 
                           for series in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, series))]

        for series_name, series_path in series_list:
            main_folder = os.path.join(series_path, main_modality)
            if not os.path.isdir(main_folder):
                continue

            informative_slice_indices = []
            for f in os.listdir(main_folder):
                if not (f.endswith(".npy") and f.startswith("slice_")):
                    continue
                try:
                    slice_idx = int(f.replace("slice_", "").replace(".npy", ""))
                except ValueError:
                    continue
                if not (min_idx <= slice_idx <= max_idx):
                    continue

                full_path = os.path.join(main_folder, f)
                if is_informative_slice(full_path, threshold=0.05):
                    informative_slice_indices.append(slice_idx)

            if not informative_slice_indices:
                continue

            # Costruisci chiave univoca
            subject_id = f"{patient}/{series_name}" if series_name else patient
            subject_dict[subject_id] = {}

            for modality in selected_modalities:
                modality_folder = os.path.join(series_path, modality)
                slice_paths = [
                    os.path.join(modality_folder, f"slice_{idx}.npy")
                    for idx in informative_slice_indices
                ]
                subject_dict[subject_id][modality] = slice_paths

    samples = []
    for subject, mod_slices in subject_dict.items():
        num_slices = len(next(iter(mod_slices.values())))
        for slice_idx in range(num_slices):
            samples.append({
                "subject": subject,
                "slice_idx": slice_idx
            })

    with open(samples_cache_path, "w") as f:
        json.dump(samples, f, indent=2)
    with open(subject_dict_cache_path, "wb") as f:
        pickle.dump(subject_dict, f)

    print(f"Cache salvata: {samples_cache_path}, {subject_dict_cache_path}")
    
    return samples, subject_dict


def split_train_test(samples, test_size=0.2):
    """
    Divide i samples in train/test in base ai subject, in modo deterministico.
    """
    # Estrai tutti i subject unici
    unique_subjects = sorted(set(s["subject"] for s in samples))  # sorted() garantisce ordine deterministico
    
    # Calcoliamo quanti vanno in test
    n_test = int(len(unique_subjects) * test_size)
    
    # Split deterministico: primi n_test subject in test, il resto in train
    test_subjects = set(unique_subjects[:n_test])
    train_subjects = set(unique_subjects[n_test:])
    
    # Ricostruiamo i samples
    train_samples = [s for s in samples if s["subject"] in train_subjects]
    test_samples  = [s for s in samples if s["subject"] in test_subjects]
    
    return train_samples, test_samples


class Mri2DSlicedataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.modalities = opt.mri_modalities
        assert len(self.modalities) >= 2, "You must specify at least two modalities (e.g., ['t1n', 't2w'])"
        self.A_mod = self.modalities[0]
        print("A_mod", self.A_mod)
        self.B_mod = self.modalities[1]
        print("B_mod", self.B_mod)
        self.data_path = opt.dataroot

        self.min_idx, self.max_idx = opt.slice_range

        self.samples, self.subject_dict = index_slices_by_subject(
            self.data_path, self.modalities, self.min_idx, self.max_idx
        )

        if opt.under_sample_dataset:
            self.samples = [x for i, x in enumerate(self.samples) if (i + 1) % 3 == 1] # (i + 1) % 3 != 0

        if opt.phase is not None: 
            train_samples, test_samples = split_train_test(self.samples, test_size=0.2)
            if opt.phase == "train":
                self.samples = train_samples
            elif opt.phase == "test":
                self.samples = test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        subject = sample_info["subject"]
        slice_idx = sample_info["slice_idx"]

        A_path = self.subject_dict[subject][self.A_mod][slice_idx]
        B_path = self.subject_dict[subject][self.B_mod][slice_idx]

        A = np.load(A_path).astype(np.float32)
        B = np.load(B_path).astype(np.float32)

        # Slices are in range 0-1, we map it to -1 1 to be coherent with TanH in CycleGAN
        A = A * 2 - 1
        B = B * 2 - 1

        A = np.expand_dims(A, axis=0)  # [1, H, W]
        B = np.expand_dims(B, axis=0)  # [1, H, W]

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
            "subject": subject,
            "slice_idx": slice_idx,
            "modality_mapping": {
                "A": self.A_mod,
                "B": self.B_mod
            }
        }


