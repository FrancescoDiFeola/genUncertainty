from __future__ import annotations

import torch

from latent_uq.losses.heteroscedastic import HeteroscedasticLoss
from latent_uq.training.generic import _compute_training_loss, _make_flow_matching_training_target


class FakeRFlowScheduler:
    def sample_timesteps(self, target: torch.Tensor) -> torch.Tensor:
        return torch.arange(target.shape[0], device=target.device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps
        return 0.25 * original_samples + 0.75 * noise


def test_rectified_flow_target_is_data_minus_noise(monkeypatch):
    target = torch.tensor([[[[2.0, 4.0]]]])
    fixed_noise = torch.tensor([[[[0.5, -1.0]]]])
    monkeypatch.setattr(torch, "randn_like", lambda _: fixed_noise.clone())

    noisy, timesteps, objective = _make_flow_matching_training_target(
        target, FakeRFlowScheduler()
    )

    assert torch.equal(timesteps, torch.tensor([0]))
    assert torch.allclose(noisy, 0.25 * target + 0.75 * fixed_noise)
    assert torch.allclose(objective, target - fixed_noise)
    assert objective.dtype == torch.float32


def test_aleatoric_flow_loss_matches_legacy_formula():
    pred_mean = torch.tensor([[[[0.2, -0.1]]]])
    pred_logvar = torch.tensor([[[[-8.0, 0.3]]]])
    target_velocity = torch.tensor([[[[1.0, -0.5]]]])
    weight = 2.5
    criterion = HeteroscedasticLoss(min_logvar=-7.0, reg_weight=1e-3)

    weighted, unweighted = _compute_training_loss(
        framework="lfm",
        mode="aleatoric",
        criterion=criterion,
        prediction=pred_mean,
        logvar=pred_logvar,
        objective=target_velocity,
        diff_loss_weight=weight,
    )

    clamped = torch.clamp(pred_logvar, min=-7.0)
    precision = torch.exp(-clamped)
    expected_unweighted = (
        0.5 * precision * (target_velocity - pred_mean) ** 2 + 0.5 * clamped
    ).mean() + 1e-3 * precision.mean()

    assert torch.allclose(unweighted, expected_unweighted)
    assert torch.allclose(weighted, weight * expected_unweighted)
