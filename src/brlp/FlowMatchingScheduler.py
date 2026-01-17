import torch

class StochasticFlowScheduler:
    """
    Stochastic Flow Matching Scheduler
    (Lipman et al. 2022, Tong et al. 2023)
    """

    def __init__(
        self,
        sigma_min=0.0,
        sigma_max=1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # -----------------------------
    # Training utilities
    # -----------------------------
    def sample_timesteps(self, batch_size, device):
        """
        Sample t ~ Uniform(0,1)
        """
        return torch.rand(batch_size, device=device)

    def sigma(self, t):
        """
        Linear noise schedule σ(t)
        """
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def sigma_dot(self, t):
        """
        Time derivative σ'(t)
        """
        return (self.sigma_max - self.sigma_min) * torch.ones_like(t)

    def interpolate(self, x0, x1, t, noise):
        """
        x_t = (1 - t) x0 + t x1 + σ(t) ε
        """
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)

        return (1 - t) * x0 + t * x1 + self.sigma(t) * noise

    def target_velocity(self, x0, x1, noise, t):
        """
        v*(x_t,t) = (x1 - x0) - σ'(t) ε
        """
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)

        return (x1 - x0) - self.sigma_dot(t) * noise

    # -----------------------------
    # Inference (probability flow ODE)
    # -----------------------------
    def step(self, x, v, dt):
        """
        Probability flow ODE step
        """
        return x + v * dt