import torch
import os


class TrainOptions:
    def __init__(self):

        self.dataroot = '/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/dataset/patches_train'


        # Training settings
        self.phase = "train"

        self.batchSize = 16

        # Optimizer settings
        self.lr = 0.0001
        self.n_epochs = 350

        # Mixed Precision
        self.amp = True

        # Loss Weights
        self.perceptual_weight = 0.3
        self.kl_weight = 1e-6
        self.adv_weight = 0.1

        # Gradient Accumulation
        self.gradient_accumulation_steps = 2 if self.batchSize < 2 else 1  # Fix tuple issue

        #self.use_cache = True