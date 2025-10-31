import os
import argparse
import warnings
import torch
from tqdm import tqdm
from monai.utils import set_determinism
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter
from src.brlp.dataset import CT2DSliceDifferenceDataset
from src.brlp import const
from src.brlp import utils
from src.brlp.latent_memory import LatentMemory
from src.brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd
)

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()
 
def supervised_contrastive_loss(latent, labels, temperature=0.1):
    """
    Computes supervised contrastive loss.
    Args:
        latent: Tensor of shape (batch_size, channels, height, width) - input embeddings.
        labels: Tensor of shape (batch_size,) - class labels.
        temperature: Temperature scaling factor.
    Returns:
        Loss value (scalar).
    """
    # Flatten latent features: (batch_size, channels, height, width) -> (batch_size, channels * height * width)
    batch_size = latent.size(0)
    latent = latent.view(batch_size, -1)

    # Normalize embeddings
    latent = F.normalize(latent, p=2, dim=1)  # Normalize along feature dimension

    # Compute pairwise cosine similarity
    cosine_similarity = torch.matmul(latent, latent.T)  # Pairwise cosine similarity
    scaled_similarity = cosine_similarity / temperature  # Scale by temperature

    # Create positive pair mask
    labels = labels.unsqueeze(1)  # (batch_size, 1)
    positive_mask = labels == labels.T  # (batch_size, batch_size)
    self_mask = torch.eye(batch_size, device=latent.device).bool()
    positive_mask = positive_mask & ~self_mask  # Exclude self-similarity

    # Compute numerator and denominator
    numerator = torch.exp(scaled_similarity) * positive_mask.float()
    numerator_sum = numerator.sum(dim=1)  # Sum over positives
    denominator = torch.logsumexp(scaled_similarity, dim=1)  # Log-sum-exp for numerical stability

    # Compute loss
    loss = -torch.log((numerator_sum + 1e-8) / torch.exp(denominator))  # Avoid log(0) by adding epsilon
    loss = loss.mean()  # Average over batch

    return loss

def latent_separation_loss(mu_2000, mu_2001):
    """Encourages latent vectors to be distinct."""
    return F.cosine_similarity(mu_2000, mu_2001).mean()  # Pushes them apart
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--cache_dir', required=False, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt', default=None, type=str)
    parser.add_argument('--disc_ckpt', default=None, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--max_batch_size', default=16, type=int)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--aug_p', default=0.8, type=float)
    args = parser.parse_args()

    # Load the LDCT/HDCT dataset
    dataset = CT2DSliceDifferenceDataset(csv_file="/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/VT/virtual_treatment_NLST.csv", global_normalization=True)
    
    
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.max_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              persistent_workers=True,
                              pin_memory=True)

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)
    # Apply Multi-GPU Training
    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs for parallel training.")
        autoencoder = torch.nn.DataParallel(autoencoder)
    
        discriminator = torch.nn.DataParallel(discriminator)
        
    adv_weight = 0.025
    perceptual_weight = 0.001
    kl_weight = 1e-7

    # Initialize latent memory storage
    latent_memory = LatentMemory()
    
    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=2,
                                      network_type="squeeze",
                                      is_fake_3d=False,
                                      ).to(DEVICE)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g,
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d,
                                     grad_scaler=GradScaler())

    avgloss = utils.AverageLoss()
    writer = SummaryWriter(comment=args.experiment_name)
    total_counter = 0

    for epoch in range(args.n_epochs):

        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')
        
        
        # Check if the memory bank is fully populated
        memory_filled = latent_memory.check_filled()
        
        for step, batch in progress_bar:

            with autocast(enabled=True):

                images = batch["difference"].to(DEVICE)
                diff_type = batch["diff_type"]
                patient_ids = batch["patient_id"]
                
                print(images.shape)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                
                # Store latents and check for pairs
                """paired_mu = []
                current_mu = []
                for i in range(len(patient_ids)):
                    patient_id = patient_ids[i].item()
                    latent_vector = mu[i]  # Now correctly using the linear output

                    # Store the current latent in memory
                    latent_memory.update(patient_id, diff_type[i], latent_vector)

                    # Try retrieving the matching latent (only after the first epoch)
                    if memory_filled:
                        paired_latent = latent_memory.get_pair(patient_id, diff_type[i])
                        if paired_latent is not None:
                            paired_mu.append(paired_latent)
                            current_mu.append(latent_vector)

                # Compute latent separation loss only when pairs exist and memory is filled
                if len(paired_mu) > 0 and memory_filled:
                    paired_mu = torch.stack(paired_mu).to(device)
                    current_mu = torch.stack(current_mu).to(device)
                    sep_loss = 1 + F.cosine_similarity(paired_mu, current_mu).mean()
                else:
                    sep_loss = torch.tensor(0.0, device=device)  # No loss applied in first epoch# No loss applied in first epoch"""
               
                
                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                

                loss_g = rec_loss + kld_loss + per_loss + gen_loss   # + sep_loss 

            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging.
            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adverarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kld_loss.item())
            # avgloss.put('Generator/separation_loss', sep_loss.item())
            avgloss.put('Discriminator/adverarial_loss', loss_d.item())
            

            if total_counter % 50 == 0:
                step = total_counter // 50
                avgloss.to_tensorboard(writer, step)
                utils.tb_display_reconstruction_2D(writer, step, images[0].detach().cpu(),
                                                reconstruction[0].detach().cpu())

            total_counter += 1
        if epoch % 50 == 0:
            # Save the model after each epoch.
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
            torch.save(autoencoder.state_dict(), os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
