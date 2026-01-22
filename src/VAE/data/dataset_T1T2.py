import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class T1T2Dataset(Dataset):

    def __init__(
        self,
        annotation_csv,
        transforms_1=None,
    ):
        self.transform1 = transforms.Compose(transforms_1) if transforms_1 is not None else None
        self.annotations = pd.read_csv(annotation_csv)
        self.dataset_len = len(self.annotations)

        print(f"Dataset size: {self.dataset_len}")

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def pad_to_256(self, image):
        pad_h = max(256 - image.shape[1], 0)
        pad_w = max(256 - image.shape[2], 0)

        padding = (
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )
        return np.pad(image, padding, mode="constant", constant_values=-1)

    # --------------------------------------------------
    # Get item
    # --------------------------------------------------
    def __getitem__(self, index):
        idx = index % self.dataset_len
        img_path = self.annotations.iloc[idx]["img_path"]

        # Load image
        item = np.load(img_path).astype(np.float32)

        # Add channel dimension
        item = np.expand_dims(item, axis=0)

        # Pad to 256x256
        item = self.pad_to_256(item)

        # Apply transforms
        if self.transform1 is not None:
            item = self.transform1(item)

        # Clip to [-1, 1]
        item = np.clip(item, -1, 1)

        # Binarization
        binarized = (item > -1).astype(np.float32)

        # Convert to tensors
        item = torch.tensor(item, dtype=torch.float32)
        binarized = torch.tensor(binarized, dtype=torch.float32)

        return {
            "img": item,                  # keep key name for compatibility
            "A_paths": img_path,
            "binarized_A": binarized,
        }

    def __len__(self):
        return self.dataset_len