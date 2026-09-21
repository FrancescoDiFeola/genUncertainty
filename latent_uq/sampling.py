"""Public sampling API and optional spatial tiling around the model-specific inference."""
import torch

from .inference import module_for
from .inference.common import Prediction
from .models import encode
from .utils import evaluating, finite


def selected_analysis(config, analysis=None):
    if analysis is not None:
        return analysis
    if len(config.inference.analyses) > 1:
        raise ValueError(
            'Select one analysis per sample() call; the CLI runs multiple analyses separately')
    return next(iter(config.inference.analyses), 'metrics')


def _full_image(condition, models, config, *, propagate, analysis='metrics'):
    inference = module_for(config.framework)
    size = config.inference.patch_size
    if size is None:
        return inference.infer(encode(models, condition, config),
                               models,
                               config,
                               analysis=analysis,
                               propagate=propagate)
    from monai.inferers import sliding_window_inference

    def predictor(window):
        result = inference.infer(encode(models, window, config),
                                 models,
                                 config,
                                 analysis=analysis,
                                 propagate=propagate)
        return torch.cat([result.image, result.variance], dim=1) if propagate else result.image

    stitched = sliding_window_inference(condition,
                                        roi_size=(size, size),
                                        sw_batch_size=config.inference.window_batch_size,
                                        predictor=predictor,
                                        overlap=config.inference.overlap,
                                        mode=config.inference.blend_mode,
                                        padding_mode='constant',
                                        cval=config.inference.pad_value)
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
