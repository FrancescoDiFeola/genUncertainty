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

    def __init__(self, annotation_A, annotation_B, window_width=1400, window_center=-400, height=256, width=256,
                 unpaired=False):
        """
        Initialize the dataset with annotations and image processing parameters.

        Parameters:
            annotation_A (str): Path to the CSV file containing image paths for domain A.
            annotation_B (str): Path to the CSV file containing image paths for domain B.
            window_width (int): Window width specifies the range of HU values to display.
            window_center (int): Center of the selected HU window.
            height (int): Height of the image after resizing.
            width (int): Width of the image after resizing.
            unpaired (bool): Whether the dataset is unpaired, which will shuffle data.
        """


        self.annotations_A = pd.read_csv(annotation_A)  # Read CSV for domain A
        self.annotations_B = pd.read_csv(annotation_B)  # Read CSV for domain B

        if unpaired:
            print("Shuffling data...")
            self.annotations_A = self.annotations_A.sample(frac=1).reset_index(drop=True)
            self.annotations_B = self.annotations_B.sample(frac=1).reset_index(drop=True)

        self.window_width = window_width
        self.window_center = window_center
        self.height = height
        self.width = width
        self.A_size = len(self.annotations_A)  # Get the size of dataset A
        self.B_size = len(self.annotations_B)  # Get the size of dataset B
        self.dataset_len = max(self.A_size, self.B_size)  # Max length between A and B

        self.plot_verbose = False

    def __getitem__(self, index):
        """Return a data point and its metadata information."""

        # Get the image path and name for domain A
        img_path_A = self.annotations_A['img_path'].iloc[index % self.A_size]
        #temp
        # img_path_A = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/L067/L067_QD_1_1.CT.0003.0001.2015.12.22.18.10.55.420810.358276339.IMA"

        img_name_A = self.annotations_A['img_name'].iloc[index % self.A_size]
        img_raw_A = pydicom.dcmread(img_path_A, force=True)  # Read the DICOM file
        img_A = self.transforms(img_raw_A)  # Apply transformations

        # Get the image path and name for domain B
        img_path_B = self.annotations_B['img_path'].iloc[index % self.B_size]

        # img_path_B = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/L067/L067_QD_1_1.CT.0003.0001.2015.12.22.18.10.55.420810.358276339.IMA"

        img_name_B = self.annotations_B['img_name'].iloc[index % self.B_size]
        img_raw_B = pydicom.dcmread(img_path_B, force=True)  # Read the DICOM file
        img_B = self.transforms(img_raw_B)  # Apply transformations

        return {'A': img_A, 'B': img_B, 'A_paths': img_path_A, 'B_paths': img_path_B}

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



if __name__ == "__main__":
     annotation_A = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/Mayo_total_ordinato_LOWDOSE.csv"
     annotation_B = "/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/Mayo_total_ordinato_LOWDOSE.csv"
     dataset = LDCTHDCTDataset(annotation_A, annotation_B)
     image = dataset[0]
     print(dataset[0]['A'].shape)
     torch.save(dataset[0]['A'], '/Users/francescodifeola/PycharmProjects/StableDiffusion-PyTorch-main_modified/data/prova_LD_tensor.pt')

     """import matplotlib
     matplotlib.use('TkAgg')
     import matplotlib.pyplot as plt

     plt.imshow(dataset[0]['A'][0, :, :])
     plt.show()"""
