# bayesdiff_model.py

import torch
import torch.nn as nn
from laplace.baselaplace import DiagLaplace
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector


class BayesDiffModel(nn.Module):
    """
    BayesDiff-style Bayesian noise predictor using
    Last-Layer Laplace Approximation (LLLA).
    """

    def __init__(self, diffusion_model, laplace_loader, device):
        super().__init__()
        self.device = device

        # -------------------------------
        # Split UNet into feature extractor + output head
        # -------------------------------
        # IMPORTANT: adapt this line if your architecture differs
        self.out_layer = diffusion_model.out[2]
        self.feature_extractor = diffusion_model
        self.feature_extractor.out[2] = nn.Identity()

        # ---------------------------
        # Laplace on last layer only
        # ---------------------------
        self.la = DiagLaplace(
            nn.Sequential(self.out_layer, nn.Flatten(1)),
            likelihood="regression",
            sigma_noise=1.0,
            prior_precision=1.0,
            backend=BackPackEF,
        )

        self.fit_laplace(laplace_loader)

    @torch.no_grad()
    def fit_laplace(self, loader):
        """
        Fit diagonal Laplace approximation on last layer.
        """
        self.la._init_H()
        self.la.model.eval()
        self.la.mean = parameters_to_vector(self.la.model.parameters()).detach()

        # infer output size
        x, t = next(iter(loader))
        x, t = x.to(self.device), t.to(self.device)
        feats = self.feature_extractor(x, t)
        out = self.la.model(feats)
        self.la.n_outputs = out.shape[-1]

        N = len(loader.dataset)

        for x, t in loader:
            x, t = x.to(self.device), t.to(self.device)
            with torch.no_grad():
                feats = self.feature_extractor(x, t)
            loss, H = self.la._curv_closure(feats, feats, None, N)
            self.la.H += H

        self.la.n_data = N

    @torch.no_grad()
    def forward(self, x, t):
        """
        Returns:
            eps_mean: (B, C, H, W)
            eps_var : (B, C, H, W)
        """
        feats = self.feature_extractor(x, t)
        eps_mean, eps_var = self.la(
            feats,
            pred_type="nn",
            link_approx="mc",
            n_samples=50,
        )
        eps_mean = eps_mean.view_as(x)
        eps_var = eps_var.view_as(x)
        return eps_mean, eps_var