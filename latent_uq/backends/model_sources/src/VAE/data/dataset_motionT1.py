import torch
import numpy as np
import pandas as pd
from torchvision import transforms


class MotionT1VaeDataset:

    def __init__(
        self,
        annotation_T1,
        mode="train",
        motion_range=(0.0, 0.15),
        fixed_motion_level=0.0,
        transforms_1=None,
        base_seed=1234,
        include_clean=True,      # whether to mix clean + corrupted
    ):
        """
        Dataset for training VAE of LDM.

        - Returns single images (no pairs).
        - Images can be clean or motion corrupted.
        - No supervision.
        """

        self.annotations = pd.read_csv(annotation_T1)
        self.size = len(self.annotations)

        self.mode = mode
        self.motion_range = motion_range
        self.fixed_motion_level = fixed_motion_level
        self.base_seed = base_seed
        self.include_clean = include_clean

        self.transform = transforms.Compose(transforms_1) if transforms_1 else None

        print(f"Dataset size: {self.size}")
        print(f"Mode: {self.mode}")

    # ============================================================
    # Motion Artifact (same as before, deterministic)
    # ============================================================
    def apply_motion_artifact(self, image, index):

        rng = np.random.default_rng(self.base_seed + index)

        if self.mode == "train":
            motion_level = rng.uniform(*self.motion_range)
        else:
            motion_level = self.fixed_motion_level

        if motion_level <= 0.0:
            return image.copy()

        img = image[0].copy()
        H, W = img.shape

        kspace = np.fft.fftshift(np.fft.fft2(img))

        num_lines = max(1, int(motion_level * H))
        lines = rng.choice(H, size=num_lines, replace=False)

        v = np.linspace(-0.5, 0.5, W)

        for line in lines:
            if line == H // 2:
                continue
            shift = rng.uniform(-10, 10)
            phase_shift = np.exp(-2j * np.pi * v * shift)
            kspace[line, :] *= phase_shift

        corrupted = np.fft.ifft2(np.fft.ifftshift(kspace))
        corrupted = np.real(corrupted)
        corrupted = np.clip(corrupted, -1.0, 1.0)

        return corrupted.astype(np.float32)[None, ...]

    # ============================================================
    def pad_to_256(self, image):
        pad_h = max(256 - image.shape[1], 0)
        pad_w = max(256 - image.shape[2], 0)

        padding = (
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )

        return np.pad(image, padding, mode="constant", constant_values=-1)

    # ============================================================
    def __getitem__(self, index):

        img_path = self.annotations.iloc[index]["img_path"]
        img = np.load(img_path).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = self.pad_to_256(img)

        # --------------------------------------------------------
        # Decide clean or corrupted
        # --------------------------------------------------------
        if self.include_clean and self.mode == "train":
            # 50% clean, 50% corrupted
            if np.random.rand() < 0.5:
                output = img.copy()
            else:
                output = self.apply_motion_artifact(img, index)
        else:
            output = self.apply_motion_artifact(img, index)

        if self.transform:
            output = self.transform(output)

        output = np.clip(output, -1, 1)
        output = torch.tensor(output, dtype=torch.float32)

        return {"img": output, "path": img_path}

    def __len__(self):
        return self.size