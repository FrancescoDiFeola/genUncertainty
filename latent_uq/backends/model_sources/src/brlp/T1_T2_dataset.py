import torch
from torchvision import transforms
import numpy as np
import pandas as pd

class T1T2Dataset():

    def __init__(self, annotation_A, annotation_B, noise_level=0, count=None, transforms_1=None, transforms_2=None, unaligned=False):  # opt
        self.transform1 = transforms.Compose(transforms_1) if transforms_1 is not None else None
        self.transform2 = transforms.Compose(transforms_2) if transforms_2 is not None else None
        self.annotations_A = pd.read_csv(annotation_A)  # legge il file annotations
        self.annotations_B = pd.read_csv(annotation_B)
        self.A_size = len(self.annotations_A)  # get the size of dataset A
        self.B_size = len(self.annotations_B)  # get the size of dataset B
        self.dataset_len = max(self.A_size, self.B_size)
        print(self.dataset_len)
        # self.files_A = sorted(glob.glob("%s/A/*" % root))
        # self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level = noise_level

    """def __getitem__(self, index):
        img_A_path = self.annotations_A.iloc[index % self.A_size]['img_path']
        img_B_path = self.annotations_B.iloc[index % self.B_size]['img_path']

        # Carica le immagini
        item_A = np.load(img_A_path).astype(np.float32)
        item_B = np.load(img_B_path).astype(np.float32)
        item_A = np.expand_dims(item_A, axis=0)  # aggiunge un canale all'inizio (axis=0)
        item_B = np.expand_dims(item_B, axis=0)
        
        # Normalize to [-1, 1] range
        # item_A = (item_A / 127.5) - 1
        # item_B = (item_B / 127.5) - 1
        # item_A = np.mean(item_A, axis=0, keepdims=True).astype(np.float32)
        # item_B = np.mean(item_B, axis=0, keepdims=True).astype(np.float32)

        # item_A = np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)
        # item_B = np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)

        if self.noise_level == 0 and self.transform2 is not None:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(item_A)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(item_B)
        else:
            if self.transform1 is not None:
                item_A = self.transform1(item_A)
                item_B = self.transform1(item_B)
        # print (item_A.ndim)
        item_A = (item_A + 1)*0.5
        item_B = (item_B + 1)*0.5
        
        return {'A': item_A, 'B': item_B, 'A_paths': img_A_path, 'B_paths': img_B_path}"""

    def add_square_artifact(self, image, value=1.0, size=32):
        _, H, W = image.shape
        top = np.random.randint(0, H - size)
        left = np.random.randint(0, W - size)
        image[:, top:top + size, left:left + size] = value  # broadcast across channel
        return image

    def __getitem__(self, index):
        img_A_path = self.annotations_A.iloc[index % self.A_size]['img_path']
        img_B_path = self.annotations_B.iloc[index % self.B_size]['img_path']

        # Load the images
        item_A = np.load(img_A_path).astype(np.float32)
        item_B = np.load(img_B_path).astype(np.float32)
        item_A = np.expand_dims(item_A, axis=0)  # Add a channel at the start (axis=0)
        item_B = np.expand_dims(item_B, axis=0)

        # Calculate padding if necessary
        def pad_to_256(image):
            pad_h = max(256 - image.shape[1], 0)
            pad_w = max(256 - image.shape[2], 0)
            # Pad equally on both sides of each dimension
            padding = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
            return np.pad(image, padding, mode='constant', constant_values=-1)

        # Pad images to 256x256 if necessary
        item_A = pad_to_256(item_A)
        # ✅ Inject square artifact to simulate OOD
        # item_A = self.add_square_artifact(item_A, value=1.0, size=32)  # adjust value and size as needed
        item_B = pad_to_256(item_B)

        # Optional transformations
        if self.noise_level == 0 and self.transform2 is not None:
            seed = np.random.randint(2147483647)  # Seed for reproducible transformations
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(item_A)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(item_B)
        else:
            if self.transform1 is not None:
                item_A = self.transform1(item_A)
                item_B = self.transform1(item_B)

        # Normalize and shift to [0, 1] range
        # item_A = (item_A + 1) * 0.5
        # item_B = (item_B + 1) * 0.5

        # Clip to ensure values are in the 0–1 range
        item_A = np.clip(item_A, -1, 1)
        item_B = np.clip(item_B, -1, 1)

        # Binarize the images: if pixel > -1, put 1; else put 0
        binarized_A = (item_A > -1).astype(np.float32)
        binarized_B = (item_B > -1).astype(np.float32)

        # Convert to tensors
        item_A = torch.tensor(item_A, dtype=torch.float32)
        item_B = torch.tensor(item_B, dtype=torch.float32)
        binarized_A = torch.tensor(binarized_A, dtype=torch.float32)
        binarized_B = torch.tensor(binarized_B, dtype=torch.float32)

        return {
            'A': item_A,
            'B': item_B,
            'A_paths': img_A_path,
            'B_paths': img_B_path,
            'binarized_A': binarized_A,
            'binarized_B': binarized_B
        }
    
    def __len__(self):
        return max(len(self.annotations_A), len(self.annotations_B))