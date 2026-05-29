import torch
import numpy as np
import pandas as pd
from torchvision import transforms


class MotionT1Dataset:

    def __init__(
        self,
        annotation_A,
        annotation_B,
        mode="train",                 # "train" or "test"
        motion_range=(0.0, 0.15),     # used in training
        fixed_motion_level=0.0,       # used in testing
        transforms_1=None,
        unaligned=False,
        base_seed=1234
    ):
        """
        mode:
            - "train": motion_level sampled uniformly in motion_range
            - "test":  motion_level fixed to fixed_motion_level

        motion_range:
            tuple (min, max) for training corruption

        fixed_motion_level:
            fixed corruption magnitude during testing

        base_seed:
            ensures deterministic corruption per sample
        """

        self.annotations_A = pd.read_csv(annotation_A)
        self.annotations_B = pd.read_csv(annotation_B)

        self.A_size = len(self.annotations_A)
        self.B_size = len(self.annotations_B)
        self.dataset_len = max(self.A_size, self.B_size)

        self.mode = mode
        self.motion_range = motion_range
        self.fixed_motion_level = fixed_motion_level
        self.base_seed = base_seed

        self.transform = transforms.Compose(transforms_1) if transforms_1 is not None else None

        self.unaligned = unaligned

        print(f"Dataset length: {self.dataset_len}")
        print(f"Mode: {self.mode}")

    # ============================================================
    # Deterministic Motion Artifact (k-space simulation)
    # ============================================================
    def apply_motion_artifact(self, image, index):
        """
        Simula artefatti da movimento nel k-space per immagini in range [-1, 1].
        """
        rng = np.random.default_rng(self.base_seed + index)

        # 1. Gestione del Motion Level
        if self.mode == "train":
            motion_level = rng.uniform(self.motion_range[0], self.motion_range[1])
        else:
            motion_level = self.fixed_motion_level

        if motion_level <= 0.0:
            return image.copy()

        # Prepariamo l'immagine: (C, H, W) -> (H, W)
        # Importante: lavoriamo su una copia per non sporcare l'originale
        img = image[0].copy()
        H, W = img.shape

        # 2. Trasformata di Fourier
        # Portiamo l'immagine nel dominio delle frequenze
        kspace = np.fft.fft2(img)
        kspace = np.fft.fftshift(kspace)

        num_lines = max(1, int(motion_level * H))
        lines = rng.choice(H, size=num_lines, replace=False)

        # Vettore delle frequenze spaziali normalizzate
        v = np.linspace(-0.5, 0.5, W)

        # 3. Applicazione del Phase Shift
        for line in lines:
            # Evitiamo di corrompere la linea DC (centro dello spazio K)
            # per preservare la luminosità media (evita il background grigio/bianco)
            if line == H // 2: continue

            shift = rng.uniform(-10, 10)
            # Teorema dello shift: f(x-Δx) <=> F(u) * exp(-2j * pi * u * Δx)
            phase_shift = np.exp(-2j * np.pi * v * shift)
            kspace[line, :] *= phase_shift

        # 4. Ricostruzione dell'immagine
        corrupted = np.fft.ifft2(np.fft.ifftshift(kspace))

        # Usiamo np.real o np.abs. Per CT in range [-1, 1], np.real è spesso
        # più fedele alla fisica del segnale originario.
        corrupted = np.real(corrupted)

        # 5. FIX CRUCIALE PER IL BACKGROUND
        # L'interferenza costruttiva/distruttiva della fase può creare pixel
        # fuori dal range [-1, 1]. Senza questo clip, il plot risulterà sbiadito.
        corrupted = np.clip(corrupted, -1.0, 1.0)

        # Ritorna con la dimensione del canale (1, H, W)
        return corrupted.astype(np.float32)[None, ...]

    # ============================================================
    # Padding helper
    # ============================================================
    def pad_to_256(self, image):
        pad_h = max(256 - image.shape[1], 0)
        pad_w = max(256 - image.shape[2], 0)

        padding = (
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        )

        return np.pad(image, padding, mode="constant", constant_values=-1)

    # ============================================================
    # GET ITEM
    # ============================================================
    def __getitem__(self, index):

        img_A_path = self.annotations_A.iloc[index % self.A_size]["img_path"]
        img_B_path = self.annotations_B.iloc[index % self.B_size]["img_path"]

        # Load images
        item_A = np.load(img_A_path).astype(np.float32)
        item_B = np.load(img_B_path).astype(np.float32)

        item_A = np.expand_dims(item_A, axis=0)
        item_B = np.expand_dims(item_B, axis=0)

        # Pad
        item_A = self.pad_to_256(item_A)
        item_B = self.pad_to_256(item_B)

        # --------------------------------------------------------
        # Clean T1
        # --------------------------------------------------------
        t1_clean = item_A.copy()

        # --------------------------------------------------------
        # Motion-corrupted T1 (deterministic per index)
        # --------------------------------------------------------
        t1_motion = self.apply_motion_artifact(t1_clean, index)

        # --------------------------------------------------------
        # Apply transforms (same to both)
        # --------------------------------------------------------
        if self.transform is not None:
            t1_clean = self.transform(t1_clean)
            t1_motion = self.transform(t1_motion)
            item_B = self.transform(item_B)

        # Clip
        t1_clean = np.clip(t1_clean, -1, 1)
        t1_motion = np.clip(t1_motion, -1, 1)
        item_B = np.clip(item_B, -1, 1)

        # Convert to tensor
        t1_clean = torch.tensor(t1_clean, dtype=torch.float32)
        t1_motion = torch.tensor(t1_motion, dtype=torch.float32)
        item_B = torch.tensor(item_B, dtype=torch.float32)

        return {
            "B": t1_clean,
            "A": t1_motion,
            "T2": item_B,
            "A_paths": img_A_path,
            "B_paths": img_B_path,
        }

    def __len__(self):
        return self.dataset_len