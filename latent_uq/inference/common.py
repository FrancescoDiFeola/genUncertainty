"""Sampling loop shared by all four frameworks, plus the Reference that parameterizes it.

Each framework's module resolves a Reference for the requested mode/analysis and calls
generate() with it. The last trajectory step is excluded from variance aggregation.
"""
from dataclasses import dataclass, replace

import torch

from ..models import predict, decode, encode_context, zero_context
from ..process import Process
from ..utils import evaluating, finite


@dataclass(frozen=True)
class Reference:
    function: str
    last_k: int = 10
    samples: int = 10
    decode_samples: int = 20
    flow_dt_squared: bool = False
    posthoc_prediction: str = "mean"
    metrics_kind: str = "reconstruction"
    calibration_bins: bool = False
    sparsification_kind: str = "fast"
    # Only ldm encodes self-conditioning context inside autocast (fp16 on CUDA);
    # dm/fm/lfm always encode it in float32.
    context_autocast: bool = False
    # Only ldm accepts context_input="prediction_variance"; dm/fm/lfm condition on
    # variance only and reject that setting.
    context_concat_allowed: bool = False


@dataclass
class Prediction:
    image: torch.Tensor
    variance: torch.Tensor | None

    @property
    def mean(self):
        """Compatibility alias; posthoc sparsification uses the FIRST image, not the mean."""
        return self.image


def resolved(config, reference):
    """Explicit overrides change defaults, never silently change the selected algorithm."""
    return replace(reference,
                   last_k=config.inference.last_k or reference.last_k,
                   samples=config.inference.samples or reference.samples,
                   decode_samples=config.inference.decode_samples or reference.decode_samples)


def trajectory(encoded, models, config, reference, process, times, *, propagate):
    channels = config.model.latent_channels if config.latent else config.model.target_channels
    state = torch.randn((len(encoded), channels, *encoded.shape[2:]),
                        dtype=encoded.dtype,
                        device=encoded.device)
    variance_sum, count = torch.zeros_like(state), 0
    previous_prediction = previous_logvar = None
    for index, timestep in enumerate(times):
        next_timestep = times[index + 1] if index + 1 < len(times) else timestep.new_zeros(())
        # The backbone always receives integer timesteps, even under the continuous
        # RFlow schedule.
        model_time = timestep.long().expand(len(state))
        model_input = torch.cat([state, encoded], dim=1)
        context = None
        amp = state.is_cuda and config.mode == 'selfcond'
        if config.mode == 'selfcond':
            if index == 0:
                with torch.autocast(device_type=state.device.type, enabled=amp):
                    previous_prediction, previous_logvar = predict(
                        models, model_input, model_time, config,
                        zero_context(len(state), config, state))
            context_logvar = previous_logvar if reference.context_autocast else previous_logvar.float(
            )
            with torch.autocast(device_type=state.device.type,
                                enabled=amp and reference.context_autocast):
                context = encode_context(models,
                                         context_logvar,
                                         config,
                                         previous_prediction,
                                         ablation=not config.inference.self_conditioning)
        with torch.autocast(device_type=state.device.type, enabled=amp):
            prediction, logvar = predict(models, model_input, model_time, config, context)
        if propagate and len(times) - reference.last_k - 1 <= index < len(times) - 1:
            factor = process.variance_factor(timestep,
                                             next_timestep,
                                             flow_dt_squared=reference.flow_dt_squared)
            variance_sum += factor * logvar.float().exp()
            count += 1
        state = process.step(prediction, state, timestep, next_timestep)
        previous_prediction, previous_logvar = prediction, logvar
    pixels = decode(models, finite(state, 'final state'), config)
    if not propagate:
        return Prediction(pixels, None)
    state_variance = finite(variance_sum / max(count, 1), 'state variance')
    if models.vae is None:
        return Prediction(pixels, state_variance)
    sigma = state_variance.clamp_min(1e-12).sqrt()
    decoded = torch.stack([
        decode(models, state + sigma * torch.randn_like(state), config)
        for _ in range(reference.decode_samples)
    ])
    return Prediction(pixels, finite(decoded.var(dim=0, unbiased=False), 'decoded variance'))


@torch.no_grad()
def generate(condition, models, config, reference, *, propagate=None):
    propagate = config.uncertainty == "propagated" if propagate is None else propagate
    if propagate and config.mode == "base":
        raise ValueError("Base models have no learned variance to propagate")
    with evaluating(*models.modules()):
        reference = resolved(config, reference)
        # ldm/lfm accept conditions that are already latent-encoded and scaled.
        encoded = condition
        process = Process(config)
        times = process.schedule(encoded.device)
        if config.uncertainty == 'posthoc':
            # One initial state is drawn and discarded, once per batch, so the random
            # sequence consumed by each of the samples below stays aligned across runs.
            channels = config.model.latent_channels if config.latent else config.model.target_channels
            torch.randn((len(encoded), channels, *encoded.shape[2:]),
                        dtype=encoded.dtype,
                        device=encoded.device)
            samples = torch.stack([
                trajectory(encoded, models, config, reference, process, times,
                           propagate=False).image for _ in range(reference.samples)
            ])
            image = samples[0] if reference.posthoc_prediction == 'first' else samples.mean(0)
            return Prediction(image, finite(samples.var(0, unbiased=False), 'posthoc variance'))
        return trajectory(encoded, models, config, reference, process, times, propagate=propagate)


def select_reference(config,
                     analysis,
                     names,
                     *,
                     last_k=10,
                     flow_dt_squared=False,
                     context_autocast=False,
                     context_concat_allowed=False):
    if analysis not in {'metrics', 'sparsification', 'calibration', 'uncertainty_summary'}:
        raise ValueError(f'Unknown inference analysis: {analysis}')
    if (config.mode == 'selfcond' and config.model.context_input != 'variance'
            and not context_concat_allowed):
        raise ValueError(f'{config.framework} conditions on variance only; '
                        'context_input="prediction_variance" is not supported here')
    shared = dict(context_autocast=context_autocast, context_concat_allowed=context_concat_allowed)
    if config.uncertainty == 'posthoc':
        if config.mode != 'base':
            raise ValueError('Post-hoc uncertainty requires a base-mode model')
        name = names['posthoc'][analysis]
        return Reference(name,
                         samples=10 if analysis == 'metrics' else 4,
                         posthoc_prediction='first' if analysis == 'sparsification' else 'mean',
                         **shared)
    if config.mode == 'base':
        if analysis != 'metrics':
            raise ValueError('Base single-trajectory inference has no uncertainty analyses')
        return Reference(names['base'], **shared)
    if not config.inference.self_conditioning and config.mode == 'selfcond':
        if analysis != 'metrics':
            raise ValueError('The selfcond ablation only supports the metrics analysis')
        return Reference(names['ablation'], last_k=last_k, **shared)
    if analysis not in names[config.mode]:
        raise ValueError(f'{config.mode} does not support the {analysis} analysis for this framework')
    return Reference(names[config.mode][analysis],
                     last_k=last_k,
                     decode_samples=10 if config.mode == 'aleatoric' else 20,
                     flow_dt_squared=flow_dt_squared,
                     **shared)
