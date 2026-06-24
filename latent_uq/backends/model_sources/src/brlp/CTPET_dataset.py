import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import random

class CTPETDataset():
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--annotation_A', type=str, default=None,
                            help='Path to the CSV file with CT and PET slice paths')
     
        return parser

    def __init__(self, opt, transform=None):
        self.data_info = pd.read_csv(opt.annotation_A)
        self.unpaired = 'paired'
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def _normalize_ct(self, ct):
        ct = np.clip(ct, -1000, 400)
        return (ct + 1000) / 1400

    def _normalize_pet(self, pet):
        return (pet - pet.min()) / (pet.max() - pet.min()) if pet.max() != pet.min() else pet

    def _pad_to_256(self, tensor):
        _, h, w = tensor.shape
        pad_h = max(0, 256 - h)
        pad_w = max(0, 256 - w)
        padding = (
            pad_w // 2, pad_w - pad_w // 2,  # left, right
            pad_h // 2, pad_h - pad_h // 2   # top, bottom
        )
        return F.pad(tensor, padding, mode='constant', value=0)

    def __getitem__(self, idx):
        row_ct = self.data_info.iloc[idx]
        row_pet = row_ct if self.unpaired=="paired" else self.data_info.sample(1).iloc[0]

        ct = np.load(row_ct['CT_Path'])
        pet = np.load(row_pet['PET_Path'])

        ct = self._normalize_ct(ct)

        ct_tensor = torch.from_numpy(ct).float().unsqueeze(0).contiguous()
        pet_tensor = torch.from_numpy(pet).float().unsqueeze(0).contiguous()

        ct_tensor = self._pad_to_256(ct_tensor)
        pet_tensor = self._pad_to_256(pet_tensor)

        sample = {
            'A': ct_tensor,
            'B': pet_tensor,
            'A_paths': row_ct['CT_Path'],
            'B_paths': row_pet['PET_Path']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample