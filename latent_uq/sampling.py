"""Public sampling API and optional spatial tiling around the model-specific inference."""
import numpy as np
import torch
from torch.nn import functional as F

from .inference import module_for
from .inference.common import Prediction, generate
from .models import encode
from .utils import evaluating, finite


def selected_analysis(config, analysis=None):
    if analysis is not None:
        return analysis
    if len(config.inference.analyses) > 1:
        raise ValueError(
            'Select one analysis per sample() call; the CLI runs multiple analyses separately')
    return next(iter(config.inference.analyses), 'metrics')


def _pad_to_window(condition, config):
    """Pad each spatial axis to at least one window, centred as MONAI's sliding window does,
    and return the crop that restores the original extent."""
    size, padding = config.inference.patch_size, []
    for extent in reversed(condition.shape[2:]):  # F.pad lists the last axis first.
        missing = max(size - extent, 0)
        padding += [missing // 2, missing - missing // 2]
    crop = (..., *[slice(max(size - extent, 0) // 2, max(size - extent, 0) // 2 + extent)
                   for extent in condition.shape[2:]])
    return F.pad(condition, padding, value=config.inference.pad_value), crop


def window_concentration(shape, config):
    """Concentration of the sliding-window blending weights at each pixel, sum(w²)/sum(w)²:
    1 where one window contributes, 1/k where k windows contribute equally. Averaging k
    independent window samples shrinks their variance by this factor."""
    from monai.data.utils import compute_importance_map, dense_patch_slices
    from monai.inferers.utils import _get_scan_interval

    size = config.inference.patch_size
    padded = tuple(max(extent, size) for extent in shape)
    roi = (size, ) * len(shape)
    weights = compute_importance_map(roi, mode=config.inference.blend_mode).double().numpy()
    total, squares = np.zeros(padded), np.zeros(padded)
    interval = _get_scan_interval(padded, roi, len(shape), (config.inference.overlap, ) * len(shape))
    for window in dense_patch_slices(padded, roi, interval):
        total[window] += weights
        squares[window] += weights**2
    crop = tuple(slice((p - e) // 2, (p - e) // 2 + e) for p, e in zip(padded, shape))
    return (squares / total**2)[crop]


def _full_image(condition, models, config, *, propagate, analysis='metrics'):
    inference = module_for(config.framework)
    size, tiling = config.inference.patch_size, config.inference.tiling
    if size is None:
        return inference.infer(encode(models, condition, config),
                               models,
                               config,
                               analysis=analysis,
                               propagate=propagate)
    if tiling == 'per_step':
        # One image-wide trajectory; the windows tile each network evaluation instead.
        padded, crop = _pad_to_window(condition, config)
        result = inference.infer(encode(models, padded, config),
                                 models,
                                 config,
                                 analysis=analysis,
                                 propagate=propagate)
        return Prediction(result.image[crop],
                          None if result.variance is None else result.variance[crop])
    from monai.inferers import sliding_window_inference
    reference = inference.reference(config, analysis)
    inputs, split = condition, None
    if tiling == 'per_window_shared_noise':
        # Every window starts from its crop of one image-wide noise draw.
        padded, crop = _pad_to_window(condition, config)
        noise = torch.randn((len(padded), config.model.target_channels, *padded.shape[2:]),
                            dtype=padded.dtype,
                            device=padded.device)
        inputs, split = torch.cat([padded, noise], dim=1), [padded.shape[1], noise.shape[1]]

    def predictor(window):
        initial = None
        if split is not None:
            window, initial = window.split(split, dim=1)
        result = generate(encode(models, window, config),
                          models,
                          config,
                          reference,
                          propagate=propagate,
                          initial=initial)
        return torch.cat([result.image, result.variance], dim=1) if propagate else result.image

    stitched = sliding_window_inference(inputs,
                                        roi_size=(size, ) * config.model.spatial_dims,
                                        sw_batch_size=config.inference.window_batch_size,
                                        predictor=predictor,
                                        overlap=config.inference.overlap,
                                        mode=config.inference.blend_mode,
                                        padding_mode='constant',
                                        cval=config.inference.pad_value)
    if split is not None:
        stitched = stitched[crop]
    if propagate:
        mean, variance = stitched.chunk(2, dim=1)
        return Prediction(mean, variance)
    return Prediction(stitched, None)


@torch.no_grad()
def sample(condition, models, config, *, analysis=None):
    """Resolve the sampling behavior for this framework/mode/analysis, then run it.

    Tiling is an extension: learned patch variances are blended spatially. For
    posthoc, independent full images are stitched before taking their variance.
    """
    config.validate()
    analysis = selected_analysis(config, analysis)
    # A custom analysis has no Reference of its own: sampling reuses whichever
    # built-in analysis its registration names (default "metrics").
    custom = config.inference.custom_analyses.get(analysis)
    sampling_analysis = custom.sampling_analysis if custom else analysis
    reference = module_for(config.framework).reference(config, sampling_analysis)
    with evaluating(*models.modules()):
        if config.uncertainty == 'posthoc' and config.inference.patch_size is not None:
            from dataclasses import replace
            from .inference.common import resolved
            plain = replace(config, inference=replace(config.inference, uncertainty='none'))
            # One initial draw is discarded here, once per full image batch, to keep the
            # random sequence aligned with the non-tiled full-image sampling below.
            channels = config.model.latent_channels if config.latent else config.model.target_channels
            from .models import encode
            encoded = encode(models, condition, config)
            torch.randn((len(encoded), channels, *encoded.shape[2:]),
                        device=encoded.device,
                        dtype=encoded.dtype)
            draws = [
                _full_image(condition, models, plain, propagate=False, analysis='metrics').image
                for _ in range(resolved(config, reference).samples)
            ]
            draws = torch.stack(draws)
            image = draws[0] if reference.posthoc_prediction == 'first' else draws.mean(0)
            result = Prediction(image, draws.var(0, unbiased=False))
        else:
            result = _full_image(condition,
                                 models,
                                 config,
                                 propagate=config.uncertainty == 'propagated',
                                 analysis=sampling_analysis)
    if result.image.shape[2:] != condition.shape[2:]:
        raise ValueError('Decoded prediction and condition must have matching spatial dimensions')
    finite(result.image, 'prediction')
    if result.variance is not None:
        finite(result.variance, 'variance')
    return result
