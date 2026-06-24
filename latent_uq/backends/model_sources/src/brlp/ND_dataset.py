import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, csv_path, root_dir="", transform_A=None, transform_B=None):
        """
        Args:
            csv_path (str): path to CSV file (no header)
            root_dir (str): root directory prepended to CSV paths
            transform_A: torchvision transform for A images
            transform_B: torchvision transform for B images
        """
        self.root_dir = root_dir
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.pairs = []
        with open(os.path.join(self.root_dir, csv_path), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 2:
                    continue
                self.pairs.append((f"{row[0]}_rgb_anon.png", f"{row[1]}_rgb_anon.png"))

        assert len(self.pairs) > 0, "CSV contains no valid image pairs"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        A_rel, B_rel = self.pairs[idx]

        A_path = os.path.join(self.root_dir, A_rel)
        B_path = os.path.join(self.root_dir, B_rel)

        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # Optional sanity check
        # assert A_img.size == (1920, 1080)
        # assert B_img.size == (1920, 1080)

        if self.transform_A:
            A_img = self.transform_A(A_img)

        if self.transform_B:
            B_img = self.transform_B(B_img)

        return {
            "A": A_img,
            "B": B_img,
            "A_path": A_rel,
            "B_path": B_rel,
        }