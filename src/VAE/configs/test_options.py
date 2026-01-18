import torch
import os


class TestOptions:
    def __init__(self):
        # Dataset paths (make configurable)
        self.dataroot = '/mimer/NOBACKUP/groups/naiss2023-6-336/lcarusone/TESI_MAGISTRALE/dataset/patches_test'




        # Training settings
        self.phase = "test"
        #self.district = "brain"  # Default district for training
        #self.batchSize = 128  # max(2, torch.cuda.device_count())  # Adjust batch size dynamically
        self.batchSize = 1

        # Optimizer settings
        self.lr = 0.0001
        self.n_epochs = 300
        self.val_interval = 50  # Reduce validation overhead

        # Mixed Precision
        self.amp = True

        # Loss Weights
        self.perceptual_weight = 0.3
        self.kl_weight = 1e-7
        self.adv_weight = 0.1
        self.contrastive_weight = 0.5

        # Gradient Accumulation
        self.gradient_accumulation_steps = 2 if self.batchSize < 2 else 1  # Fix tuple issue

        #self.use_cache = True