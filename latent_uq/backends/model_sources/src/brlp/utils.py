from typing import Union
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from monai import transforms
from monai.data.meta_tensor import MetaTensor
from torch.utils.tensorboard.writer import SummaryWriter


class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}

    def put(self, loss_key: str, loss_value: Union[int, float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)

    def pop_avg(self, loss_key: str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)

    def to_tensorboard(self, writer: SummaryWriter, step: int):
        """
        Logs the average value of all the metrics stored 
        into Tensorboard.

        Args:
            writer (SummaryWriter): Tensorboard writer
            step (int): Tensorboard logging global step 
        """
        for metric_key in self.losses_accumulator.keys():
            writer.add_scalar(metric_key, self.pop_avg(metric_key), step)


def to_vae_latent_trick(z: torch.Tensor, unpadded_z_shape: tuple = (3, 15, 18, 15)) -> torch.Tensor:
    """
    The latent for the VAE is not divisible by 4 (required to
    go through the UNet), therefore we apply padding before using 
    it with the UNet. This function removes the padding.

    Args:
        z (torch.Tensor): Padded latent
        unpadded_z_shape (tuple, optional): unpadded latent dimensions. Defaults to (3, 15, 18, 15).

    Returns:
        torch.Tensor: Latent without padding
    """
    padder = transforms.DivisiblePad(k=4)
    z = padder(MetaTensor(torch.zeros(unpadded_z_shape))) + z
    z = padder.inverse(z)
    return z


def to_mni_space_1p5mm_trick(x: torch.Tensor, mni1p5_dim: tuple = (122, 146, 122)) -> torch.Tensor:
    """
    The volume is resized to be divisible by 8 (required by 
    the autoencoder). This function restores the initial dimensions
    (i.e., the MNI152 space dimensions at 1.5 mm^3). 

    Args:
        x (torch.Tensor): Resized volume
        mni1p5_dim (tuple, optional): MNI152 space dims at 1.5 mm^3. Defaults to (122, 146, 122).

    Returns:
        torch.Tensor: input resized to original shape
    """
    resizer = transforms.ResizeWithPadOrCrop(spatial_size=mni1p5_dim, mode='minimum')
    return resizer(x)


def tb_display_reconstruction(writer, step, image, recon):
    """
    Display reconstruction in TensorBoard during AE training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if len(image.shape) == 4: image = image.squeeze(0)
    if len(recon.shape) == 4: recon = recon.squeeze(0)

    ax[0, 0].set_title('original image', color='cyan')
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title('reconstructed image', color='magenta')
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    writer.add_figure('Reconstruction', plt.gcf(), global_step=step)

def tb_display_reconstruction_2D(writer, step, image, recon, label):
    """
    Display reconstruction in TensorBoard during AE training for 2D images.
    Args:
        writer: TensorBoard writer.
        step: Training step for logging.
        image: Original image (2D tensor).
        recon: Reconstructed image (2D tensor).
        label: Unique identifier for different autoencoders ("Diff_AE" or "Init_AE").
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 5))  # Single-row layout for clarity

    for _ax in ax:
        _ax.set_axis_off()

    # Convert to 2D if needed
    if len(image.shape) == 3:  # Shape (C, H, W)
        image = image[0]  # Assuming single-channel grayscale
    if len(recon.shape) == 3:
        recon = recon[0]  

    ax[0].set_title('Original Image', color='cyan')
    ax[0].imshow(image, cmap='gray')

    ax[1].set_title('Reconstructed Image', color='magenta')
    ax[1].imshow(recon, cmap='gray')

    difference = image - recon  # Difference Image
    ax[2].set_title('Difference', color='yellow')
    ax[2].imshow(difference, cmap='gray')

    plt.tight_layout()
    writer.add_figure(f'Reconstruction_{label}', fig, global_step=step)  # Unique name per AE
    plt.close(fig)  # Avoid memory leaks


def tb_display_generation(writer, step, tag, image):
    """
    Display generation result in TensorBoard during Diffusion Model training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, figsize=(7, 3))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)


def tb_display_generation_v2(writer, step, tag, image_1, image_2, image_gen, image_gen_2):
    """
    Display generation results of three images (image_1, image_2, and image_gen)
    in TensorBoard during Diffusion Model training, with titles for each.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=4, figsize=(7, 4))

    # Turn off axis and set titles for each subplot
    titles = ["T1", "T2", "T2 gen", "T2_rec"]
    images = [image_1.cpu().detach(), image_2.cpu().detach(), image_gen.cpu().detach(), image_gen_2.cpu().detach()]
    for i, (title, img) in enumerate(zip(titles, images)):
        print(img.shape)
        ax[i].set_axis_off()
        ax[i].set_title(title)  # Add title for each image
        if img.ndim == 2:  # Grayscale
            ax[i].imshow(img, cmap='gray')
        elif img.ndim == 3:  # RGB
            ax[i].imshow(img)
        elif img.ndim == 4:
            ax[i].imshow(img[0, 0, :, :], cmap='gray')
        else:
            raise ValueError("Unsupported image dimensions. Expected 2D grayscale or 3D RGB image.")

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.close()  # Close the figure after logging to avoid memory issues

def tb_display_generation_time_differences(writer, step, tag, 
                                           image_1, image_2, image_3, 
                                           image_4, image_5, image_6, 
                                           image_7, image_8, image_9):
    """
    Display generation results for multiple time points in TensorBoard during 
    Diffusion Model training, with titles for each.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        step (int): Training step/epoch.
        tag (str): Tag for TensorBoard logging.
        image_1, image_2, ..., image_9: Images to be displayed.
    """
    plt.style.use('dark_background')
    
    # Define titles and images
    titles = [
        "T0 (Scan_1999)", "T1 (Scan_2000)", "T2 (Scan_2001)", 
        "Diff T0->T1", "Predicted Diff T0->T1", "Predicted T1",
        "Diff T0->T2", "Predicted Diff T0->T2", "Predicted T2"
    ]
    
    images = [
        image_1.cpu().detach(), image_2.cpu().detach(), image_3.cpu().detach(),
        image_4.cpu().detach(), image_5.cpu().detach(), image_6.cpu().detach(),
        image_7.cpu().detach(), image_8.cpu().detach(), image_9.cpu().detach()
    ]
    
    # Create figure with 9 subplots
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

    for i, (title, img) in enumerate(zip(titles, images)):
        row, col = divmod(i, 3)  # Get row and column index
        ax[row, col].set_axis_off()
        ax[row, col].set_title(title, fontsize=10)

        # Handle grayscale or RGB images
        if img.ndim == 2:  # Grayscale
            ax[row, col].imshow(img, cmap='gray')
        elif img.ndim == 3:  # RGB
            ax[row, col].imshow(img)
        elif img.ndim == 4:  # Batch dimension exists
            ax[row, col].imshow(img[0, 0, :, :], cmap='gray')
        else:
            raise ValueError("Unsupported image dimensions. Expected 2D grayscale or 3D RGB image.")

    plt.tight_layout()
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)  # Close the figure after logging to free memory

def tb_display_cond_generation(writer, step, tag, starting_image, followup_image, predicted_image):
    """
    Display conditional generation result in TensorBoard during ControlNet training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0, 0].set_title('starting image', color='cyan')
    ax[0, 0].imshow(starting_image[starting_image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(starting_image[:, starting_image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(starting_image[:, :, starting_image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title('follow-up image', color='magenta')
    ax[1, 0].imshow(followup_image[followup_image.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(followup_image[:, followup_image.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(followup_image[:, :, followup_image.shape[2] // 2], cmap='gray')

    ax[2, 0].set_title('predicted follow-up', color='yellow')
    ax[2, 0].imshow(predicted_image[predicted_image.shape[0] // 2, :, :], cmap='gray')
    ax[2, 1].imshow(predicted_image[:, predicted_image.shape[1] // 2, :], cmap='gray')
    ax[2, 2].imshow(predicted_image[:, :, predicted_image.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)


def percnorm_nifti(mri, lperc=1, uperc=99):
    '''
    Apply percnorm to NiFTI1Image class
    '''
    norm_arr = percnorm(mri.get_fdata(), lperc, uperc)
    return nib.Nifti1Image(norm_arr, mri.affine, mri.header)


def percnorm(arr, lperc=1, uperc=99):
    '''
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    '''
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def apply_mask(mri, segm):
    """
    Performs brain extraction.
    """
    segm = resample_from_to(segm, mri, order=0)
    mask = segm.get_fdata() > 0
    mri_arr = mri.get_fdata()
    mri_arr[mask == 0] = 0
    return nib.Nifti1Image(mri_arr, mri.affine, mri.header)

def charbonnier_loss(pred, target, epsilon=1e-3):
    """
    Compute the Charbonnier loss between the predicted and target tensors.
    
    Args:
    - pred (torch.Tensor): The predicted tensor (e.g., noise_pred in your case).
    - target (torch.Tensor): The target tensor (e.g., noise in your case).
    - epsilon (float): A small constant for numerical stability, default is 1e-3.
    
    Returns:
    - torch.Tensor: The mean Charbonnier loss across the batch.
    """
    diff = pred - target
    loss = torch.sqrt(diff ** 2 + epsilon ** 2)
    return loss.mean()
