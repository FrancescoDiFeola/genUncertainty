import torch
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os

class CT2DSliceDifferenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, global_normalization=True, transform=None):
        """
        Args:
            csv_file (str): CSV file with paths to difference images.
            global_normalization (bool): If True, compute mean/std on the entire dataset.
                                         If False, compute mean/std per image dynamically.
            transform (callable, optional): Optional transform (e.g., augmentations).
        """
        self.data_info = pd.read_csv(csv_file)

        self.transform = transform
        self.global_normalization = global_normalization

        if self.global_normalization:
            self.mean_D, self.std_D = self._compute_difference_mean_std()
        else:
            self.mean_D, self.std_D = None, None  # Will compute per-image mean/std on-the-fly

        # Precompute valid slice indices for all patients
        self.slice_indices = self._prepare_slices()

    def _compute_difference_mean_std(self, old_str=None, new_str=None):
        """Compute dataset-wide mean and standard deviation for all difference images,
        with optional path string replacement, using an efficient online algorithm.
        """

        # Strip column names to remove any unwanted spaces
        self.data_info.columns = self.data_info.columns.str.strip()

        # Print available column names for debugging
        print("Available columns in CSV:", self.data_info.columns.tolist())

        n_total_voxels = 0  # Running total of voxel count
        mean_accum = 0.0  # Running mean
        m2_accum = 0.0  # Running sum of squared differences for variance computation

        for _, row in self.data_info.iterrows():
            print(f"Row ID: {row.name}")  # Print row index to verify iteration

            for diff_col in ['Diff_2000', 'Diff_2001']:  # Iterate over both difference columns
                if diff_col in self.data_info.columns:  # Ensure column exists
                    diff_path = row[diff_col]

                    # Perform path replacement if old_str and new_str are provided
                    if old_str and new_str:
                        diff_path = diff_path.replace(old_str, new_str)

                    if isinstance(diff_path, str) and os.path.exists(diff_path):  # Ensure valid file path
                        diff_image = sitk.ReadImage(diff_path)
                        diff_array = sitk.GetArrayFromImage(diff_image).flatten()  # Flatten to 1D array

                        n_voxels = len(diff_array)
                        if n_voxels == 0:
                            continue  # Skip empty images

                        # Compute mean & variance incrementally using Welford's algorithm
                        delta = diff_array - mean_accum
                        mean_accum += np.sum(delta) / (n_total_voxels + n_voxels)
                        m2_accum += np.sum(delta * (diff_array - mean_accum))
                        n_total_voxels += n_voxels

                    else:
                        print(f"Warning: Missing or invalid path for {diff_col} in row {row.name}: {diff_path}")
                else:
                    print(f"Error: Column {diff_col} not found in CSV!")

        # Compute final mean and standard deviation
        if n_total_voxels > 1:
            mean_D = mean_accum
            std_D = np.sqrt(m2_accum / (n_total_voxels - 1))  # Unbiased std dev
        else:
            print("Error: No valid difference images found!")
            mean_D, std_D = 0, 1  # Fallback values to avoid crashing

        print(f"Computed dataset-wide mean: {mean_D:.4f}, std: {std_D:.4f}")
        return mean_D, std_D

    def _prepare_slices(self):
        """Precompute valid slice indices for each patient, treating `diff_2000` and `diff_2001` separately."""
        slice_indices = []
        for idx, row in self.data_info.iterrows():
            scan_path = row['Diff_2000']
            image = sitk.ReadImage(scan_path)
            depth = image.GetSize()[2]

            for slice_idx in range(depth):
                # Treat each slice of Diff_2000 and Diff_2001 as separate samples
                slice_indices.append((idx, slice_idx, 'Diff_2000'))
                slice_indices.append((idx, slice_idx, 'Diff_2001'))

        return slice_indices

    def _normalize_difference_z_global(self, image):
        """Apply dataset-wide Z-score normalization to difference images."""
        return (image - self.mean_D) / self.std_D

    def _normalize_difference_z_per_image(self, image):
        """Apply per-image Z-score normalization to difference images."""
        mean_D = np.mean(image)
        std_D = np.std(image) + 1e-8  # Avoid division by zero
        return (image - mean_D) / std_D
    
    def _normalize_scan_1999(self, image):
        """Normalize `Scan_1999` using a fixed HU range [-1024, 1000]."""
        min_HU, max_HU = -1024, 1000
        norm_image = (image - min_HU) / (max_HU - min_HU)
        norm_image = np.clip(norm_image, 0, 1)  # Ensure values are in [0,1]
        norm_image = 2 * norm_image - 1  # Scale to [-1,1]
        return norm_image
    
    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        patient_idx, slice_idx, diff_col = self.slice_indices[idx]
        row = self.data_info.iloc[patient_idx]

        # Load the chosen difference image (either Diff_2000 or Diff_2001)
        diff_image = sitk.GetArrayFromImage(sitk.ReadImage(row[diff_col]))
        diff_2d = diff_image[slice_idx]

        # Apply either global or per-image Z-score normalization
        if self.global_normalization:
            diff_2d = self._normalize_difference_z_global(diff_2d)
        else:
            diff_2d = self._normalize_difference_z_per_image(diff_2d)


        # Load the initial scan (Scan_1999)
        scan_1999_image = sitk.GetArrayFromImage(sitk.ReadImage(row["Scan_1999"]))
        scan_1999 = scan_1999_image[slice_idx]
        scan_1999 = self._normalize_scan_1999(scan_1999)
        
        # Load the initial scan (Scan_2000)
        scan_2000_image = sitk.GetArrayFromImage(sitk.ReadImage(row["Scan_2000"]))
        scan_2000 = scan_2000_image[slice_idx]
        scan_2000 = self._normalize_scan_1999(scan_2000)
                
        # Load the initial scan (Scan_2001)
        scan_2001_image = sitk.GetArrayFromImage(sitk.ReadImage(row["Scan_2001"]))
        scan_2001 = scan_2001_image[slice_idx]
        scan_2001 = self._normalize_scan_1999(scan_2001)
        
        # Load the initial scan (Diff_2000)
        diff_2000_image = sitk.GetArrayFromImage(sitk.ReadImage(row["Diff_2000"]))
        diff_2000 = diff_2000_image[slice_idx]
        
        # Load the initial scan (Diff_2001)
        diff_2001_image = sitk.GetArrayFromImage(sitk.ReadImage(row["Diff_2001"]))
        diff_2001 = diff_2001_image[slice_idx]
        
        # Apply either global or per-image Z-score normalization
        if self.global_normalization:
            diff_2000 = self._normalize_difference_z_global(diff_2000)
            diff_2001 = self._normalize_difference_z_global(diff_2001)
        else:
            diff_2000 = self._normalize_difference_z_per_image(diff_2000)
            diff_2001 = self._normalize_difference_z_per_image(diff_2001)


        # Convert to PyTorch tensor and add channel dimension (1, H, W)
        diff_2d = torch.tensor(diff_2d, dtype=torch.float32).unsqueeze(0)
        scan_1999 = torch.tensor(scan_1999, dtype=torch.float32).unsqueeze(0)
        scan_2000 = torch.tensor(scan_2000, dtype=torch.float32).unsqueeze(0)
        scan_2001 = torch.tensor(scan_2001, dtype=torch.float32).unsqueeze(0)
        diff_2000 = torch.tensor(diff_2000, dtype=torch.float32).unsqueeze(0)
        diff_2001 = torch.tensor(diff_2001, dtype=torch.float32).unsqueeze(0)
        
        return {
            "difference": diff_2d,
            "scan_1999": scan_1999,
            "scan_2000": scan_2000,
            "scan_2001": scan_2001,
            "diff_2000": diff_2000, 
            "diff_2001": diff_2001,  
            "patient_id": patient_idx,
            "slice_idx": slice_idx,
            "diff_type": diff_col,
            "mean_D": self.mean_D,
            "std_D": self.std_D,
        }



if __name__ == "__main__":
    dataset = CT2DSliceDifferenceDataset(csv_file="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/virtual_treatment_NLST.csv", global_normalization=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        print("Batch keys:", batch.keys())  # Output dictionary keys
        print("Diff 2000 shape:", batch["diff_2000"].shape)  # Expected (8, 1, H, W)