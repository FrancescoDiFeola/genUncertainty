import torch
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Resized,
)
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, MapTransform
from typing import Mapping, Hashable, Any, Dict




params = {
    'WINDOW_WIDTH': 70,
    'WINDOW_LEVEL': 30,
    'num_pool': 100, #number of images generated in image pool
    'roi_size': [64, 64, 64], #determines the patch size
    #'pixdim':(0.86, 0.86, 2.50), #resampling pixel dimensions
    'imgA_intensity_range': (-1000, 1000), #range of intensities for nomalization to range [0,1]
    'imgB_intensity_range': (-1000, 1000)
}
#print("ROI:",params['roi_size'])

# QUESTO è PER IL PRETRAIN

train_transforms = Compose([

    LoadImaged(keys=['image', 'mask']),
    EnsureChannelFirstd(keys=['image', 'mask']),

    # Normalizza l'intensità dei pixel delle immagini in un determinato intervallo
    #ScaleIntensityRanged(keys=['A'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=0, b_max=1.0, clip=True),
    #RandSpatialCropd(keys=['A'], roi_size=params['roi_size'], random_size=False, random_center=True),
    #SpatialPadd(keys=['A'], spatial_size=params['roi_size']),


    #RandFlipd(keys=['A'], spatial_axis=[0], prob=0.3),
    #RandFlipd(keys=['A'], spatial_axis=[1], prob=0.3),
    #RandRotated(keys=['A'], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.3),
    #RandGaussianNoised(keys=['A'], prob=0.1, mean=0.0, std=0.05),
    #RandAdjustContrastd(keys=['A'], prob=0.1, gamma=(0.7, 1.3)),  # valori di gamma, 1 è identità, quindi 0.7 un pò più luminosa, 1.3 un po più scuri


    # SE VOGLIO APPLICARE IL RUMORE E CONTRASTO SU UNA SOLO DELLE DUE IMMAGINI, COSI DA GENERALIZZARE UNA SCANZIONE DIVERSA IN BASELINE O FOLLOWUP.

])

test_transforms = Compose([

    LoadImaged(keys=['image', 'mask']),
    EnsureChannelFirstd(keys=['image', 'mask']),

    #ScaleIntensityRanged(keys=['A'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=0, b_max=1.0, clip=True),


])


