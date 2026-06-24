import pandas as pd
import pydicom
import numpy as np
import torch
import cv2


class LDCTHDCTDataset():
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A and domain B,
    along with their annotations, during training and testing.
    """

    @staticmethod
    def convert_in_hu(dicom_file):
        """
        Apply a linear transformation to convert pixel values to HU values
        using the DICOM metadata (intercept and slope).
        """
        image = dicom_file.pixel_array
        intercept = dicom_file.RescaleIntercept
        slope = dicom_file.RescaleSlope
        image = slope * image + intercept
        return image  # Return the image in HU units

    @staticmethod
    def normalize_img(x, lower=None, upper=None, data_range='-11'):
        """
        Normalize the image x to either [0, 1] or [-1, 1] depending on data_range.
        """
        if np.max(x) != np.min(x):
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))  # Map between 0 and 1
        if data_range == '01':
            return x_norm
        else:
            return (2 * x_norm) - 1  # Map between -1 and 1

    def __init__(self, annotation, window_width=1400, window_center=-400, height=256, width=256):
        """
        Initialize the dataset with annotations and image processing parameters.

        Parameters:
            annotation (str): Path to the CSV file containing image paths.

            window_width (int): Window width specifies the range of HU values to display.
            window_center (int): Center of the selected HU window.
            height (int): Height of the image after resizing.
            width (int): Width of the image after resizing.
            unpaired (bool): Whether the dataset is unpaired, which will shuffle data.
        """

        self.annotations = pd.read_csv(annotation)  # Read CSV for domain A

        self.window_width = window_width
        self.window_center = window_center
        self.height = height
        self.width = width
        self.dataset_len = len(self.annotations)  # Get the size of dataset A

        self.plot_verbose = False

    def __getitem__(self, index):
        """Return a data point and its metadata information."""

        # Get the image path and name for domain A
        img_path = self.annotations['img_path'].iloc[index % self.dataset_len]
        # temp
        # img_path_A = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/L067/L067_QD_1_1.CT.0003.0001.2015.12.22.18.10.55.420810.358276339.IMA"

        img_name = self.annotations['img_name'].iloc[index % self.dataset_len]
        img_raw = pydicom.dcmread(img_path, force=True)  # Read the DICOM file
        img = self.transforms(img_raw)  # Apply transformations

        return {'A': img, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_len

    def window_image(self, hu_img):
        """
        Select the display window based on window center and window width.
        """
        img_w = hu_img.copy()
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        img_w[img_w < img_min] = img_min
        img_w[img_w > img_max] = img_max
        return img_w  # Return the image within the specified window

    def transforms(self, dicom, tensor_output=True):
        """
        Apply preprocessing to the DICOM image: convert to HU, apply windowing, normalize, and resize.
        """
        x = self.convert_in_hu(dicom)  # Convert the image to HU
        # x = self.window_image(x)  # Apply the selected window
        # x = self.normalize_img(x)  # Normalize the image
        x = self.lumTrans(x)
        x = cv2.resize(x, (self.height, self.width))  # Resize the image

        if tensor_output:
            x = torch.from_numpy(x)  # Convert to torch tensor
            x = x.unsqueeze(dim=0)  # Add an additional dimension
            return x.float()
        else:
            return x.astype('float32')

    def lumTrans(self, img):
        lungwin = np.array([-1024., 600])  # lung 600, totale 3071 emphysema: -400
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        # plt.imshow(newimg[200, :, :], cmap="gray")
        # plt.show()
        # plt.close()
        # newimg = (newimg * 255).astype('uint8')
        return newimg  # 2*newimg - 1
