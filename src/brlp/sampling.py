import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm
import torch.nn.functional as F
from . import utils
from . import const
from . import util_losses
from inferers import DiffusionInferer



def denormalize_image(image, method="range", min_HU=-1024, max_HU=1000, mean_HU=-1024, std_HU=600):
    """
    Denormalize an image that was normalized using either:
    - "range" method: Converts [-1,1] back to the original HU range [min_HU, max_HU]
    - "standard" method: Converts zero-mean standard deviation normalized values back to HU scale

    Args:
        image (numpy array or torch tensor): Normalized image
        method (str): "range" for min-max scaling, "standard" for mean/std normalization
        min_HU (int): Minimum HU value used in min-max scaling
        max_HU (int): Maximum HU value used in min-max scaling
        mean_HU (int): Mean HU value used in standard normalization
        std_HU (int): Standard deviation used in standard normalization

    Returns:
        denormalized_image (numpy array or torch tensor): Restored HU values
    """
    if method == "range":
        # Convert [-1,1] to [0,1]
        original_range = (image + 1) / 2
        # Convert [0,1] to the original HU range
        denormalized_image = original_range * (max_HU - min_HU) + min_HU

    elif method == "standard":
        # Convert standard normalized values back to HU range
        denormalized_image = image * std_HU + mean_HU

    else:
        raise ValueError("Invalid method. Choose either 'range' or 'standard'.")

    return denormalized_image

@torch.no_grad()
def sample_using_diffusion_unpaired(  # faccio il sampling per l'approccio unpaired
    nets,
    src_image: torch.Tensor,
    trg_image: torch.Tensor,
    diffusion: nn.Module,
    device: str,
    num_training_steps: int = 1000,
    num_inference_fdp: int = 70,
    num_inference_rdp: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205,
) -> torch.Tensor: 

    src_image = src_image[0, :, :, :].unsqueeze(0)
    trg_image = trg_image[0, :, :, :].unsqueeze(0)
    # the subject-specific variables and the progression-related
    # covariates are concatenated into a vector outside this function.
    src_latents, _ = nets.encode(src_image)
    trg_latents, _ = nets.encode(trg_image)

    trg_like_latents = util_losses.AdaIN(src_latents, trg_latents)

    trg_like_latents = ((trg_like_latents - trg_like_latents.mean()) / trg_like_latents.std()).float()
    trg_latents_std = ((trg_latents - trg_latents.mean()) / trg_latents.std()).float()

    # Using DDIM sampling from (Song et al., 2020) allowing for a
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)
                              
    inferer = DiffusionInferer(scheduler=scheduler)

    # FDP
    scheduler.set_timesteps(num_inference_steps=num_inference_fdp)
    z_noisy = inferer.reverse_sample(input_noise=trg_like_latents, diffusion_model=diffusion, scheduler=scheduler, conditioning=trg_latents_std, mode='concat', verbose=False)
    print(f"z_noisy: {z_noisy.shape}")

    # RDP
    scheduler.set_timesteps(num_inference_steps=num_inference_rdp)
    z_denoised= inferer.sample(
        input_noise=z_noisy, diffusion_model=diffusion, scheduler=scheduler,
        conditioning=trg_latents_std, mode='concat')
    print(f"z_denoised: {z_denoised.shape}")

    # Standardizzazione
    z_denoised = (z_denoised - z_denoised.mean()) / z_denoised.std()

    x = nets.decode_stage_2_outputs(z_denoised.to(device)).float()
    x_from_real = nets.decode_stage_2_outputs(trg_latents.to(device)).float()

    return x, x_from_real

@torch.no_grad()
def sampling_difference(  # faccio il sampling per l'approccio unpaired
    autoencoder_diff,
    autoencoder_init,
    diff_t1: torch.Tensor,
    diff_t2: torch.Tensor,
    img_t0: torch.Tensor,
    img_t1: torch.Tensor,
    img_t2: torch.Tensor,
    mean,
    std,
    diffusion: nn.Module,
    DEVICE: str,
    num_training_steps: int = 1000,
    num_inference_fdp: int = 70,
    num_inference_rdp: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205,
) -> torch.Tensor: 

    diff_t1 = diff_t1[0, :, :, :].unsqueeze(0)
    diff_t2 = diff_t2[0, :, :, :].unsqueeze(0)
    img_t0 = img_t0[0, :, :, :].unsqueeze(0)
    img_t1 = img_t1[0, :, :, :].unsqueeze(0)
    img_t2 = img_t2[0, :, :, :].unsqueeze(0)
    
    img_t0_latent, _ = autoencoder_init.module.encode(img_t0)
    print(img_t0_latent.shape)
    # Convert it to a PyTorch tensor
    time_interval_1 = torch.tensor(0, dtype=torch.float32, device=DEVICE)  # Shape: [1]
    time_interval_2 = torch.tensor(1, dtype=torch.float32, device=DEVICE)  # Shape: [1]
   
    # **Ensure `time_interval` has the correct shape**
    time_interval_1 = time_interval_1.view(1, 1, 1, 1)  # Shape: [1, 1, 1, 1]
    time_interval_1 = time_interval_1.expand(1, 1, img_t0_latent.shape[2], img_t0_latent.shape[3])  # Shape: [B, 1, H, W]
    
    # **Ensure `time_interval` has the correct shape**
    time_interval_2 = time_interval_2.view(1, 1, 1, 1)  # Shape: [1, 1, 1, 1]
    time_interval_2 = time_interval_2.expand(1, 1, img_t0_latent.shape[2], img_t0_latent.shape[3])  # Shape: [B, 1, H, W]

    # **Concatenate with `init_latent`**
    combined_condition_1 = torch.cat([img_t0_latent, time_interval_1], dim=1)  # Shape: [B, C+1, H, W]
    combined_condition_2 = torch.cat([img_t0_latent, time_interval_2], dim=1)  # Shape: [B, C+1, H, W]



    # Using DDIM sampling from (Song et al., 2020) allowing for a
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)
                              
    inferer = DiffusionInferer(scheduler=scheduler)
    
    # drawing a random z_T ~ N(0,I)
    z_1 = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(DEVICE).float()
    
    # RDP
    scheduler.set_timesteps(num_inference_steps=num_inference_rdp)
    z_denoised_1= inferer.sample(
        input_noise=z_1, diffusion_model=diffusion, scheduler=scheduler,
        conditioning=combined_condition_1, mode='concat')
    print(f"z_denoised: {z_denoised_1.shape}")

    # Standardizzazione
    z_denoised_1 = (z_denoised_1 - z_denoised_1.mean()) / z_denoised_1.std()
    print(mean, std)
    diff_t1_pred = autoencoder_diff.module.decode_stage_2_outputs(z_denoised_1.to(DEVICE)).float()
    diff_t1_pred_deno = denormalize_image(diff_t1_pred, method="standard", mean_HU=mean, std_HU=std)
    img_t0_deno = denormalize_image(img_t0, method="range")
    
    img_t1_pred = img_t0_deno + diff_t1_pred_deno
    
    
    z_2 = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(DEVICE).float()

    # RDP
    z_denoised_2 = inferer.sample(
        input_noise=z_2, diffusion_model=diffusion, scheduler=scheduler,
        conditioning=combined_condition_2, mode='concat')
    print(f"z_denoised: {z_denoised_2.shape}")
    
    epsilon = 1e-6
    # Standardizzazione
    z_denoised_2 = (z_denoised_2 - z_denoised_2.mean()) / (z_denoised_2.std() + epsilon)

    diff_t2_pred = autoencoder_diff.module.decode_stage_2_outputs(z_denoised_2.to(DEVICE)).float()
    diff_t2_pred_deno = denormalize_image(diff_t2_pred, method="standard", mean_HU=mean, std_HU=std)
    
    
    img_t2_pred = img_t0_deno + diff_t2_pred_deno
    
    return img_t1_pred, img_t2_pred, diff_t1_pred_deno, diff_t2_pred_deno



@torch.no_grad()
def sample_using_diffusion(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        context: torch.Tensor,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).to(device).to(device)

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                timestep = torch.tensor([t]).to(device)

                # predict the noise
                noise_pred = diffusion(
                    x=z.float(),
                    timesteps=timestep,
                    context=context.float(),
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)

    # decode the latent
    z = (z / scale_factor).float()
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x


@torch.no_grad()
def sample_using_diffusion_v2(
        autoencoder1: nn.Module,
        autoencoder2: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.

    img_1 = img_1[0, :, :, :].unsqueeze(0)
    img_2 = img_2[0, :, :, :].unsqueeze(0)
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    latents_1, _ = autoencoder1.encode(img_1)  # .module
    latents_2, _ = autoencoder2.encode(img_2)
    # latents_1 = F.pad(latents_1, pad=(1, 1, 1, 1), mode='constant', value=0)
    
    context = latents_1  # torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    # context = context.repeat(1, 3, 1, 1).float()
    
    # context = torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    # context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device).float()

    z_input = torch.concat([z, context], dim=1).float()

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            # with autocast(enabled=True):

            timestep = torch.tensor([t]).to(device)

            # predict the noise
            noise_pred = diffusion(
                x=z_input.float(),
                timesteps=timestep,
                # context=context.float(),
            )

            # the scheduler applies the formula to get the
            # denoised step z_{t-1} from z_t and the predicted noise
            z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = ((z.float()-z.mean())/z.std()).float()
    latents_2_estimate = (latents_1.float() + z).float()  # z
    # latents_2_estimate = latents_2_estimate[:, :, 1:-1, 1:-1]
    #  z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    x = autoencoder2.decode_stage_2_outputs(latents_2_estimate.to(device)).float()  # z.unsqueeze(0).to(device)
    x_from_real = autoencoder2.decode_stage_2_outputs(latents_2.to(device)).float()
    #  x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x, x_from_real

# funzione per calcolare reconstruction loss durante il training
def sample_using_diffusion_v3(
        autoencoder1: nn.Module,
        autoencoder2: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    batch_size = img_1.shape[0]
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    latents_1, _ = autoencoder1.module.encode(img_1)
    latents_2, _ = autoencoder2.module.encode(img_2)
    # latents_1 = F.pad(latents_1, pad=(1, 1, 1, 1), mode='constant', value=0)

    context = torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(batch_size, * const.LATENT_SHAPE_DM).to(device).float()  # 16, *
    # print(z.shape)
    z_input = torch.concat([z, context], dim=1).float()
    # print(z_input.shape)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        # with autocast(enabled=True):
        timestep = torch.tensor([t] * batch_size).to(device)
        # timestep = torch.tensor([t]).to(device)

        # predict the noise
        noise_pred = diffusion(
            x=z_input.float(),
            timesteps=timestep,
            # context=context.float(),
        )

        # the scheduler applies the formula to get the
        # denoised step z_{t-1} from z_t and the predicted noise
        z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = (z.float() / scale_factor).float()
    latents_2_estimate = (latents_1.float() + z.float()).float()
    # latents_2_estimate = latents_2_estimate[:, :, 1:-1, 1:-1]
    # z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    # x = autoencoder2.module.decode_stage_2_outputs(latents_2_estimate.to(device)).float()  # z.unsqueeze(0).to(device)
    # x_from_real = autoencoder2.decode_stage_2_outputs(latents_2.to(device)).float()
    # x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return latents_2_estimate

# funzione per calcolare reconstruction loss durante il training
def sample_using_diffusion_v3_(
        autoencoder1: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    batch_size = img_1.shape[0]
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    latents_1, _ = autoencoder1.encode(img_1)  # .module
    latents_2, _ = autoencoder1.encode(img_2)  # .module
    # latents_1 = F.pad(latents_1, pad=(1, 1, 1, 1), mode='constant', value=0)

    context = torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(batch_size, * const.LATENT_SHAPE_DM).to(device).float()  # 16, *
    # print(z.shape)
    z_input = torch.concat([z, context], dim=1).float()
    # print(z_input.shape)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        # with autocast(enabled=True):
        timestep = torch.tensor([t] * batch_size).to(device)
        # timestep = torch.tensor([t]).to(device)

        # predict the noise
        noise_pred = diffusion(
            x=z_input.float(),
            timesteps=timestep,
            # context=context.float(),
        )

        # the scheduler applies the formula to get the
        # denoised step z_{t-1} from z_t and the predicted noise
        z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = (z.float() / scale_factor).float()
    latents_2_estimate = (latents_1.float() + z.float()).float()
    # latents_2_estimate = latents_2_estimate[:, :, 1:-1, 1:-1]
    # z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    # x = autoencoder2.module.decode_stage_2_outputs(latents_2_estimate.to(device)).float()  # z.unsqueeze(0).to(device)
    # x_from_real = autoencoder2.decode_stage_2_outputs(latents_2.to(device)).float()
    # x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return latents_2_estimate

# this function perform sampling using a single autoencoder for both modalities
@torch.no_grad()
def sample_using_diffusion_v4_latent_difference(
        autoencoder1: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.

    img_1 = img_1[0, :, :, :].unsqueeze(0)
    img_2 = img_2[0, :, :, :].unsqueeze(0)



    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    latents_1, _ = autoencoder1.encode(img_1)  # .module
    latents_2, _ = autoencoder1.encode(img_2)

    context = latents_1  # torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    # context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device).float()

    z_input = torch.concat([z, context], dim=1).float()

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            # with autocast(enabled=True):

            timestep = torch.tensor([t]).to(device)

            # predict the noise
            noise_pred = diffusion(
                x=z_input.float(),
                timesteps=timestep,
                # context=context.float(),
            )

            # the scheduler applies the formula to get the
            # denoised step z_{t-1} from z_t and the predicted noise
            z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = ((z.float()-z.mean())/z.std()).float()
    latents_2_estimate = (latents_1.float() + z).float()  # z  # (latents_1.float() + z).float()

    x = autoencoder1.decode_stage_2_outputs(latents_2_estimate.to(device)).float()  # latents_2_estimate.to(device)

    x_from_real = autoencoder1.decode_stage_2_outputs(latents_2.to(device)).float()

    return x, x_from_real

# this function perform sampling using a single autoencoder for both modalities
@torch.no_grad()
def sample_using_diffusion_v4(
        autoencoder1: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.

    img_1 = img_1[0, :, :, :].unsqueeze(0)
    img_2 = img_2[0, :, :, :].unsqueeze(0)
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    latents_1, _ = autoencoder1.encode(img_1)  # .module
    latents_2, _ = autoencoder1.encode(img_2)

    context = latents_1  # torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    # context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device).float()

    z_input = torch.concat([z, context], dim=1).float()

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            # with autocast(enabled=True):

            timestep = torch.tensor([t]).to(device)

            # predict the noise
            noise_pred = diffusion(
                x=z_input.float(),
                timesteps=timestep,
                # context=context.float(),
            )

            # the scheduler applies the formula to get the
            # denoised step z_{t-1} from z_t and the predicted noise
            z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = ((z.float()-z.mean())/z.std()).float()
    latents_2_estimate = z 

    x = autoencoder1.decode_stage_2_outputs(latents_2_estimate.to(device)).float()  # latents_2_estimate.to(device)

    x_from_real = autoencoder1.decode_stage_2_outputs(latents_2.to(device)).float()

    return x, x_from_real



# this function perform sampling using a single autoencoder for both modalities
@torch.no_grad()
def sample_using_diffusion_test(
        autoencoder1: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    img_1 = img_1[0, :, :, :].unsqueeze(0)
    img_2 = img_2[0, :, :, :].unsqueeze(0)
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    latents_1, _ = autoencoder1.encode(img_1)  # .module
    latents_2, _ = autoencoder1.encode(img_2)

    context = latents_1  # torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    # context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device).float()

    z_input = torch.concat([z, context], dim=1).float()

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            # with autocast(enabled=True):

            timestep = torch.tensor([t]).to(device)

            # predict the noise
            noise_pred = diffusion(
                x=z_input.float(),
                timesteps=timestep,
                # context=context.float(),
            )

            # the scheduler applies the formula to get the
            # denoised step z_{t-1} from z_t and the predicted noise
            z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = (z.float() / scale_factor).float()

    latents_2_estimate = z.float()  #  (latents_1.float() + z).float()

    x = autoencoder1.decode_stage_2_outputs(latents_2_estimate.to(device)).float()  # latents_2_estimate.to(device)

    x_from_real = autoencoder1.decode_stage_2_outputs(latents_2.to(device)).float()

    return x, x_from_real

# this function perform sampling using a single autoencoder for both modalities
@torch.no_grad()
def sample_using_diffusion_test_get_latents(
        autoencoder1: nn.Module,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        diffusion: nn.Module,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    img_1 = img_1[0, :, :, :].unsqueeze(0)
    img_2 = img_2[0, :, :, :].unsqueeze(0)
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related
    # covariates are concatenated into a vector outside this function.
    latents_1, _ = autoencoder1.encode(img_1)  # .module
    latents_2, _ = autoencoder1.encode(img_2)

    context = latents_1  # torch.nn.functional.interpolate(img_1, size=latents_1.shape[-2:])
    # context = context.repeat(1, 3, 1, 1).float()

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device).float()

    z_input = torch.concat([z, context], dim=1).float()

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            # with autocast(enabled=True):

            timestep = torch.tensor([t]).to(device)

            # predict the noise
            noise_pred = diffusion(
                x=z_input.float(),
                timesteps=timestep,
                # context=context.float(),
            )

            # the scheduler applies the formula to get the
            # denoised step z_{t-1} from z_t and the predicted noise
            z, _ = scheduler.step(noise_pred.float(), t, z)

    # decode the latent
    z = (z.float() / scale_factor).float()

    latents_2_estimate = z  # (latents_1.float() + z).float()

    return latents_1, latents_2, z, latents_2_estimate

@torch.no_grad()
def sample_using_controlnet_and_z(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        controlnet: nn.Module,
        starting_z: torch.Tensor,
        starting_a: int,
        context: torch.Tensor,
        device: str,
        scale_factor: int = 1,
        average_over_n: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """
    The inference process described in the paper.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        controlnet (nn.Module): the ControlNet
        starting_z (torch.Tensor): the latent from the MRI of the starting visit 
        starting_a (int): the starting age
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        average_over_n (int, optional): LAS parameter m. Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # preparing controlnet spatial condition.
    starting_z = starting_z.unsqueeze(0).to(device)
    concatenating_age = torch.tensor([starting_a]).view(1, 1, 1, 1, 1).expand(1, 1, *starting_z.shape[-3:]).to(device)
    controlnet_condition = torch.cat([starting_z, concatenating_age], dim=1).to(device)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).unsqueeze(0).to(device)

    # if performing LAS, we repeat the inputs for the diffusion process
    # m times (as specified in the paper) and perform the reverse diffusion
    # process in parallel to avoid overheads.
    if average_over_n > 1:
        context = context.repeat(average_over_n, 1, 1)
        controlnet_condition = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1)

        # this is z_T - the starting noise.
    z = torch.randn(average_over_n, *starting_z.shape[1:]).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                # convert the timestep to a tensor.
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                # get the intermediate features from the ControlNet
                # by feeding the starting latent, the covariates and the timestep
                down_h, mid_h = controlnet(
                    x=z.float(),
                    timesteps=timestep,
                    context=context,
                    controlnet_cond=controlnet_condition.float()
                )

                # the diffusion takes the intermediate features and predicts
                # the noise. This is why we conceptualize the two networks as
                # as a unified network.
                noise_pred = diffusion(
                    x=z.float(),
                    timesteps=timestep,
                    context=context.float(),
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)

    # Here we conclude Latent Average Stabilization by averaging 
    # m different latents from m different samplings.
    z = (z / scale_factor).sum(axis=0) / average_over_n
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())

    # decode the latent using the Decoder block from the KL autoencoder.
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x
