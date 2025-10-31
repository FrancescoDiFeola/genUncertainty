import pandas as pd
import numpy as np
import pydicom
import torch
import cv2
from torchvision import transforms


class CombinedDataset:
    def __init__(
            self,
            annotation_A,
            annotation_B,
            noise_level=0,
            count=None,
            transforms_1=None,
            transforms_2=None,
            unaligned=False,
            window_width=1400,
            window_center=-400,
            height=256,
            width=256,
            tensor_output=True,
    ):
        self.transform1 = transforms.Compose(transforms_1) if transforms_1 else None
        self.transform2 = transforms.Compose(transforms_2) if transforms_2 else None
        self.annotations_A = pd.read_csv(annotation_A)
        self.annotations_B = pd.read_csv(annotation_B)

        if unaligned:
            self.annotations_A = self.annotations_A.sample(frac=1).reset_index(drop=True)
            self.annotations_B = self.annotations_B.sample(frac=1).reset_index(drop=True)

        self.noise_level = noise_level
        self.unaligned = unaligned
        self.window_width = window_width
        self.window_center = window_center
        self.height = height
        self.width = width
        self.tensor_output = tensor_output

        self.A_size = len(self.annotations_A)
        self.B_size = len(self.annotations_B)
        self.dataset_len = max(self.A_size, self.B_size)

    def __len__(self):
        return self.dataset_len

    @staticmethod
    def convert_in_hu(dicom_file):
        image = dicom_file.pixel_array
        intercept = dicom_file.RescaleIntercept
        slope = dicom_file.RescaleSlope
        return slope * image + intercept

    @staticmethod
    def normalize_img(x, lower=None, upper=None, data_range='-11'):
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else x
        return x_norm if data_range == '01' else (2 * x_norm) - 1

    def window_image(self, hu_img):
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        hu_img[hu_img < img_min] = img_min
        hu_img[hu_img > img_max] = img_max
        return hu_img

    def lumTrans(self, img):
        lungwin = np.array([-1024., 600])
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        return newimg

    def dicom_transforms(self, dicom):
        x = self.convert_in_hu(dicom)
        x = self.lumTrans(x)
        x = cv2.resize(x, (self.height, self.width))
        if self.tensor_output:
            x = torch.from_numpy(x).unsqueeze(dim=0).float()
        return x

    def pad_to_256(self, image):
        pad_h = max(256 - image.shape[1], 0)
        pad_w = max(256 - image.shape[2], 0)
        padding = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        return np.pad(image, padding, mode='constant', constant_values=-1)

    def __getitem__(self, index):
        img_A_path = self.annotations_A.iloc[index % self.A_size]['img_path']
        img_B_path = self.annotations_B.iloc[index % self.B_size]['img_path']

        try:
            img_raw_A = pydicom.dcmread(img_A_path, force=True)
            item_A = self.dicom_transforms(img_raw_A)
        except Exception:
            item_A = np.load(img_A_path).astype(np.float32)
            item_A = np.expand_dims(item_A, axis=0)
            item_A = self.pad_to_256(item_A)
            item_A = torch.tensor(item_A, dtype=torch.float32)

        try:
            img_raw_B = pydicom.dcmread(img_B_path, force=True)
            item_B = self.dicom_transforms(img_raw_B)
        except Exception:
            item_B = np.load(img_B_path).astype(np.float32)
            item_B = np.expand_dims(item_B, axis=0)
            item_B = self.pad_to_256(item_B)
            item_B = torch.tensor(item_B, dtype=torch.float32)

        if self.noise_level == 0 and self.transform2 is not None:
            seed = np.random.randint(2147483647)
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

        binarized_A = (item_A > -1)
        binarized_B = (item_B > -1)
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




if __name__ == "__main__":
    annotation_A = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/Mayo_total_ordinato_LOWDOSE.csv"
    annotation_B = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/Mayo_total_ordinato_LOWDOSE.csv"
    dataset = LDCTHDCTDataset(annotation_A, annotation_B)
    image = dataset[0]
    print(dataset[0]['A'].shape)
    torch.save(dataset[0]['A'],
               '/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/prova_LD_tensor.pt')

    """import matplotlib
     matplotlib.use('TkAgg')
     import matplotlib.pyplot as plt

     plt.imshow(dataset[0]['A'][0, :, :])
     plt.show()"""
