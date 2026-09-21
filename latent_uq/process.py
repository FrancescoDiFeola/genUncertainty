"""Diffusion/flow schedulers, and the noisy states and training targets they produce."""
import torch
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.networks.schedulers import RFlowScheduler


class Process:

    def __init__(self, config):
        self.config = config
        p = config.process
        if config.diffusion:
            options = dict(num_train_timesteps=p.num_train_timesteps,
                           beta_start=p.beta_start,
                           beta_end=p.beta_end,
                           schedule=p.beta_schedule)
            self.train_scheduler = DDPMScheduler(**options)
            self.sample_scheduler = DDIMScheduler(**options, clip_sample=False)
        else:
            base = p.flow_base_size or (64**2 if config.latent else 256**2)
            # lfm/aleatoric inference uses 256², unlike lfm training and every other lfm mode.
            sample_base = p.flow_inference_base_size or (256**2 if config.framework == 'lfm'
                                                         and config.mode == 'aleatoric' else base)
            options = dict(num_train_timesteps=p.num_train_timesteps,
                           use_discrete_timesteps=False,
                           sample_method='uniform',
                           use_timestep_transform=True,
                           spatial_dim=2)
            self.train_scheduler = RFlowScheduler(**options, base_img_size_numel=base)
            self.sample_scheduler = RFlowScheduler(**options, base_img_size_numel=sample_base)
            self.input_size = p.flow_inference_size or sample_base

    def training_pair(self, target, *, noise=None, timesteps=None):
        # Noise is drawn before timesteps in both branches, so a fixed seed always
        # reproduces the same pair regardless of framework.
        noise = torch.randn_like(target) if noise is None else noise
        if self.config.diffusion:
            times = torch.randint(self.config.process.num_train_timesteps, (len(target), ),
                                  device=target.device) if timesteps is None else timesteps
            objective = noise
        else:
            times = self.train_scheduler.sample_timesteps(
                target) if timesteps is None else timesteps
            objective = target - noise
        state = self.train_scheduler.add_noise(target, noise, times)
        return state, times, objective

    def schedule(self, device):
        if self.config.diffusion:
            self.sample_scheduler.set_timesteps(self.config.steps)
            self.sample_scheduler.alphas_cumprod = self.sample_scheduler.alphas_cumprod.to(device)
        else:
            self.sample_scheduler.set_timesteps(self.config.steps,
                                                device=device,
                                                input_img_size_numel=self.input_size)
        return self.sample_scheduler.timesteps.to(device)

    def step(self, prediction, state, timestep, next_timestep):
        if self.config.diffusion:
            return self.sample_scheduler.step(prediction, timestep, state, eta=0.0)[0]
        return self.sample_scheduler.step(prediction, timestep, state, next_timestep)[0]

    def variance_factor(self, timestep, next_timestep=None, *, flow_dt_squared=False):
        if self.config.diffusion:
            alpha = self.sample_scheduler.alphas_cumprod[timestep.long()]
            return (1 - alpha) / (alpha + 1e-8)
        # See Reference.flow_dt_squared for which fm analyses set this to True.
        return 1 / self.config.steps**2 if flow_dt_squared else 1.0


def heteroscedastic_loss(prediction, logvar, target, regularization=0.001, min_logvar=-7.0):
    """Gaussian negative log-likelihood with a log-variance floor and a precision penalty.

    The floor applies only to this loss, not to the context encoder or to inference.
    """
    logvar = logvar.clamp_min(min_logvar)
    precision = torch.exp(-logvar)
    return (0.5 * precision * (target - prediction).square() +
            0.5 * logvar).mean() + regularization * precision.mean()
