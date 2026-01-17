import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CityscapesColorDataset(Dataset):
    """
    Returns (image, semantic_color_map) pairs from Cityscapes.
    Only *_gtFine_color.png is used as ground truth.
    """

    def __init__(self, root, split="train", transform=None, target_transform=None):
        """
        Args:
            root (str or Path): Cityscapes root directory
            split (str): 'train', 'val', or 'test'
            transform: transform applied to the RGB image
            target_transform: transform applied to the color semantic map
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.img_dir = self.root / "leftImg8bit" / split
        self.gt_dir = self.root / "gtFine" / split

        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []

        for city_dir in sorted(self.img_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            city = city_dir.name
            gt_city_dir = self.gt_dir / city

            for img_path in city_dir.glob("*_leftImg8bit.png"):
                # Build corresponding gtFine_color path
                gt_name = img_path.name.replace(
                    "_leftImg8bit.png", "_gtFine_color.png"
                )
                gt_path = gt_city_dir / gt_name

                if gt_path.exists():
                    samples.append((img_path, gt_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        target = Image.open(gt_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"A": image, "B": target}