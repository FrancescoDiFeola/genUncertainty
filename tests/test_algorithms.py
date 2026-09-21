from dataclasses import replace
import math

import pytest
import torch

from latent_uq.models import Models, build_models, encode, decode
from latent_uq.process import Process, heteroscedastic_loss
from latent_uq.sampling import sample, Prediction
from latent_uq.training import train_step


def test_loss_matches_gaussian_likelihood_and_analytic_gradients():
    prediction = torch.tensor([0.3, -1.0], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.2, 0.5], dtype=torch.float64)
    logvar = torch.tensor([-0.4, 0.7], dtype=torch.float64, requires_grad=True)
    reg = 0.001
    loss = heteroscedastic_loss(prediction, logvar, target, reg)
    normal = torch.distributions.Normal(prediction, (logvar / 2).exp())
    reference = (-normal.log_prob(target) - 0.5 * math.log(2 * math.pi) + reg *
                 (-logvar).exp()).mean()
    torch.testing.assert_close(loss, reference)
    loss.backward()
    torch.testing.assert_close(prediction.grad,
                               (prediction.detach() - target) * (-logvar.detach()).exp() / 2)
    expected = (0.5 - (0.5 * (prediction.detach() - target)**2 + reg) *
                (-logvar.detach()).exp()) / 2
    torch.testing.assert_close(logvar.grad, expected)


@pytest.mark.parametrize("framework", ["dm", "fm", "ldm", "lfm"])
def test_noising_and_supervision(config_factory, framework):
    config = config_factory(framework)
    process = Process(config)
    data = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    noise = torch.tensor([-2.0, -3.0]).reshape_as(data)
    times = torch.tensor([0, 500])
    noisy, returned, objective = process.training_pair(data, noise=noise, timesteps=times)
    if config.diffusion:
        alpha = process.train_scheduler.alphas_cumprod[times].reshape_as(data)
        torch.testing.assert_close(noisy, alpha.sqrt() * data + (1 - alpha).sqrt() * noise)
        torch.testing.assert_close(objective, noise)
    else:
        torch.testing.assert_close(noisy, torch.tensor([1.0, -0.5]).reshape_as(data))
        torch.testing.assert_close(objective, data - noise)
    torch.testing.assert_close(returned, times)


@pytest.mark.parametrize("framework", ["dm", "fm", "ldm", "lfm"])
def test_selfcond_training_preserves_framework_specific_state_and_encoder(
        config_factory, framework):
    config = config_factory(framework)
    models = build_models(config)
    loss = train_step(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), models, Process(config),
                      config)
    loss.backward()
    first, second = models.backbone.calls
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert first[3] is False and second[3] is True
    assert not first[2].any()
    assert second[2].abs().sum() > 0
    from latent_uq.models import norm_percentile
    torch.testing.assert_close(
        models.context.inputs[0],
        norm_percentile(models.backbone.logvar.detach().expand_as(models.context.inputs[0]).exp()))
    assert sum(float(p.grad.abs().sum()) for p in models.context.parameters()) > 0
    if models.vae:
        assert not any(p.requires_grad or p.grad is not None for p in models.vae.parameters())


@pytest.mark.parametrize("framework", ["dm", "fm"])
def test_sampler_extra_initial_pass_and_final_excluded_factor(config_factory, framework):
    config = config_factory(framework)
    models = build_models(config)
    result = sample(torch.zeros(2, 1, 8, 8), models, config)
    assert len(models.backbone.calls) == config.steps + 1
    assert not models.backbone.calls[0][2].any()
    assert len(models.context.inputs) == config.steps
    schedule = Process(config)
    times = schedule.schedule("cpu")
    if config.diffusion:
        alpha = schedule.sample_scheduler.alphas_cumprod[times[-3:-1]]
        factor = ((1 - alpha) / (alpha + 1e-8)).mean()
    else:
        factor = torch.tensor(1 / config.steps**2)
    torch.testing.assert_close(
        result.variance,
        models.backbone.logvar.detach().exp().expand_as(result.variance) * factor)
    assert models.backbone.training  # Sampling restores model state.


def test_ablation_encodes_zero_maps_and_omits_flow_variance_factor(config_factory):
    config = config_factory("fm")
    config.inference.self_conditioning = False
    models = build_models(config)
    result = sample(torch.zeros(1, 1, 8, 8), models, config)
    assert not models.backbone.calls[0][2].any()
    assert len(models.context.inputs) == config.steps
    assert all(not value.any() for value in models.context.inputs)
    torch.testing.assert_close(result.variance, torch.full_like(result.variance, math.exp(-1)))


def test_flow_steps_retain_reference_timestep_truncation(config_factory):
    config = config_factory("fm")
    config.inference.steps = 30
    process = Process(config)
    times = process.schedule("cpu")
    state = torch.tensor([3.0])
    for i, t in enumerate(times):
        next_t = times[i + 1] if i + 1 < len(times) else torch.tensor(0.)
        state = process.step(torch.tensor([-5.0]), state, t, next_t)
    # MONAI 1.5.2 casts next_timestep to int, so the integrated interval is >1.
    assert state.item() == pytest.approx(3 - 5 * 1.01015625, abs=2e-6)


def test_posthoc_aggregates_complete_stitched_images(config_factory, monkeypatch):
    config = config_factory("dm", "base")
    config.inference.uncertainty = "posthoc"
    config.inference.patch_size = 4
    values = [torch.full((1, 1, 8, 8), v) for v in (1., 2., 6.)]
    call = []

    def full_image(*args, propagate, analysis):
        assert propagate is False
        call.append(len(call))
        return Prediction(values[len(call) - 1].clone(), None)

    monkeypatch.setattr("latent_uq.sampling._full_image", full_image)
    result = sample(torch.zeros_like(values[0]), build_models(config), config)
    assert len(call) == 3
    torch.testing.assert_close(result.mean, torch.stack(values).mean(0))
    torch.testing.assert_close(result.variance, torch.stack(values).var(0, unbiased=False))


def test_latent_channel_variance_and_scaling(config_factory):
    config = config_factory("lfm")
    config.inference.decode_samples = 2500
    config.inference.self_conditioning = False
    models = build_models(config)
    image = torch.ones(1, 1, 8, 8)
    torch.testing.assert_close(encode(models, image, config)[:, 0], image[:, 0] * 2.5)
    # Use a linear decoder whose exact propagated variance is known for unequal channels.
    models.vae.decode = lambda z: 2 * z[:, :1] + 3 * z[:, 1:2]
    result = sample(image, models, config)
    expected = (4 * math.exp(-1) + 9) / 2.5**2
    assert abs(float(result.variance.mean()) - expected) / expected < 0.03


@pytest.mark.parametrize("uncertainty", ["propagated", "posthoc", "none"])
def test_patch_full_image_shapes_and_single_window_equivalence(config_factory, uncertainty):
    config = config_factory("fm")
    if uncertainty == 'posthoc':
        config = config_factory('fm', 'base')
    config.inference.uncertainty = uncertainty
    condition = torch.rand(2, 1, 8, 8)
    models = build_models(config)
    torch.manual_seed(15)
    full = sample(condition, models, config)
    config.inference.patch_size = 8
    config.inference.window_batch_size = 2
    torch.manual_seed(15)
    patch = sample(condition, models, config)
    torch.testing.assert_close(full.mean, patch.mean)
    if full.variance is not None:
        torch.testing.assert_close(full.variance, patch.variance)
    config.inference.patch_size = 4
    overlapping = sample(condition, models, config)
    assert overlapping.mean.shape == condition.shape
    assert torch.isfinite(overlapping.mean).all()
