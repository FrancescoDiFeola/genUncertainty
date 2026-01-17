import torch.nn as nn
import torch.nn.functional as F
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
import torch


class VAE_Losses:

    def kl_loss(self, mu, log_var):

        z_sigma = torch.exp(0.5 * log_var)
        eps = 1e-10

        kl_loss = 0.5 * torch.sum(
            mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
            dim=list(range(1, len(z_sigma.shape))),
        )
        return torch.sum(kl_loss) / kl_loss.shape[0]


    def recon_loss_weighted(self, reconstruction, image, mask):
        
        recon = torch.abs(reconstruction - image)
        weights = 1.0 + (1.0 * mask) 
   
        recon_loss = (recon * weights).mean()

        return recon_loss


    def __init__(self, device, perceptual_weight=0.3, kl_weight=1e-6, adv_weight=0.1):

        #self.recon_loss = nn.L1Loss()  # Reconstruction loss (L1)
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2, pretrained="DEFAULT").to(device)


        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight



    def compute_losses(self, reconstruction, image, mu, log_var, discriminator, mask):


        losses = {
            "recon": self.recon_loss_weighted(reconstruction, image, mask),  # Latent consistency loss
            "kl": self.kl_loss(mu, log_var),   # KL loss
            "perceptual": self.perceptual_loss(reconstruction, image),  # Perceptual loss
        }

        # Adversarial loss: Discourage trivial latent encodings by making z_ct and z_pet realistic
        logits_fake = discriminator(reconstruction.float())[-1]
        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

         # Total adversarial loss
        generator_loss_tot = (generator_loss) * 0.5

        # Final weighted loss for the encoder
        loss_g = (
            losses["recon"]+
            self.kl_weight * losses["kl"] +  # Ensure smooth latent representation
            self.perceptual_weight * losses["perceptual"] +  # Maintain high-level structure
            self.adv_weight * generator_loss_tot    # Adversarial regularization

        )

        return losses, loss_g
