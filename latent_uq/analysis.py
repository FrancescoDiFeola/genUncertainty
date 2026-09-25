"""Reconstruction, sparsification, calibration and uncertainty-summary analyses.

Reconstruction/summary metrics use channel zero; sparsification uses every channel.
Sample IDs and undefined statistics are handled explicitly.
"""
from contextlib import ExitStack
from pathlib import Path
import csv

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .inference import module_for
from .models import norm_percentile
from .utils import import_object

SUMMARY = ['mae', 'u_mean', 'u_p95', 'u_p99', 'u_top1_mean', 'u_top5_mean']
CORRELATIONS = ['pearson', 'spearman', 'auroc_top5', 'auroc_top10', 'auroc_top15']
SCHEMAS = {
    'metrics':
    ['mse', 'psnr', 'ssim', *CORRELATIONS, *[f'{n}_norm' for n in CORRELATIONS], *SUMMARY],
    'sparsification': ['fraction', 'error', 'random_error', 'oracle_error'],
    'calibration': ['bin', 'p_low', 'p_high', 'mean_uncertainty', 'mean_error', 'count'],
    'uncertainty_summary': SUMMARY,
    'calibration_bins': ['bin', 'mean_uncertainty', 'mean_error', 'count'],
    'window_profile': [
        'bin', 'concentration_low', 'concentration_high', 'mean_error', 'mean_uncertainty',
        'count'
    ],
}


class CustomAnalysis:
    """Base for a pluggable inference analysis registered via inference.custom_analyses.

    Unlike the four built-in analyses, a custom analysis has no numerical regression
    test of its own: it is a project-specific extension. Sampling (which produces
    target/prediction/uncertainty) is still driven by the built-in analysis named in
    the registration's sampling_analysis field; a subclass only defines what to do
    with that output.
    """
    columns: list[str] = []
    requires_uncertainty: bool = False

    def __call__(self, target, prediction, uncertainty) -> list[dict]:
        """target, prediction: (C,H,W) or (C,D,H,W) numpy arrays. uncertainty: same shape or None.
        When the dataset returns regions, a boolean `mask` over the spatial axes is also
        passed as a keyword argument, once per region.

        Returns one dict per output row, with exactly the keys in `columns`.
        """
        raise NotImplementedError


def _curve(error, order, fractions, fast=True):
    sorted_error = error[order]
    if not fast:
        return np.asarray([sorted_error[int(f * len(error)):].mean() for f in fractions])
    removed = np.minimum(np.round(fractions * len(error)).astype(int), len(error) - 1)
    cumulative = np.cumsum(sorted_error)
    remaining = np.where(removed == 0, cumulative[-1], cumulative[-1] - cumulative[removed - 1])
    return remaining / (len(error) - removed)


def sparsification(uncertainty, error, *, kind='fast'):
    """Reference curves: fast rounds counts/20 trials; standard floors counts/10 trials."""
    u, e = np.asarray(uncertainty).ravel(), np.asarray(error).ravel()
    if not len(e):
        raise ValueError('Sparsification requires at least one pixel')
    fractions = np.linspace(0, .95, 50)
    fast = kind == 'fast'
    if kind not in {'fast', 'standard'}:
        raise ValueError('sparsification kind must be fast or standard')
    curve = _curve(e, np.argsort(-u), fractions, fast)
    oracle = _curve(e, np.argsort(-e), fractions, fast)
    trials = 20 if fast else 10
    random = sum((_curve(e, np.random.permutation(len(e)), fractions, fast)
                  for _ in range(trials)), np.zeros_like(fractions)) / trials

    def normalize(value):
        denominator = value[0] + (1e-12 if fast else 0)
        return value / denominator if denominator else np.zeros_like(value)

    return fractions, normalize(curve), normalize(random), normalize(oracle)


def sparsification_scores(fractions, curve, random_curve, oracle):
    """AUSE (curve vs. oracle) and AURG (random vs. curve): areas under sparsification()'s curves."""
    return dict(ause=float(trapezoid(curve - oracle, fractions)),
                aurg=float(trapezoid(random_curve - curve, fractions)))


def _summary(uncertainty, error):
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if not u.size:
        return dict.fromkeys(SUMMARY)
    ordered = np.sort(u)
    return dict(mae=float(np.mean(error)),
                u_mean=float(u.mean()),
                u_p95=float(np.percentile(u, 95)),
                u_p99=float(np.percentile(u, 99)),
                u_top1_mean=float(ordered[-max(1, int(.01 * len(u))):].mean()),
                u_top5_mean=float(ordered[-max(1, int(.05 * len(u))):].mean()))


def _correlations(uncertainty, error):
    u, e = uncertainty.ravel(), error.ravel()
    result = dict(pearson=None, spearman=None)
    if len(u) > 1 and np.ptp(u) > 0 and np.ptp(e) > 0:
        result.update(pearson=float(pearsonr(u, e)[0]), spearman=float(spearmanr(u, e)[0]))
    for top in (5, 10, 15):
        labels = e > np.percentile(e, 100 - top)
        result[f'auroc_top{top}'] = float(roc_auc_score(labels,
                                                        u)) if np.unique(labels).size == 2 else None
    return result


def _tail_bins(uncertainty, error):
    u, e = uncertainty.ravel(), error.ravel()
    percentiles = [0, 50, 75, 90, 95, 99, 100]
    edges = np.percentile(u, percentiles)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (u >= low) & (u <= high if index == len(edges) - 2 else u < high)
        if mask.any():
            rows.append(
                dict(bin=index,
                     p_low=percentiles[index],
                     p_high=percentiles[index + 1],
                     mean_uncertainty=float(u[mask].mean()),
                     mean_error=float(e[mask].mean()),
                     count=int(mask.sum())))
    return rows


def _calibration_bins(uncertainty, error):
    u = torch.from_numpy(uncertainty.copy()).flatten()
    e = torch.from_numpy(error.copy()).flatten()
    edges = torch.quantile(u, torch.linspace(0, 1, 16))
    ids = torch.bucketize(u, edges[1:-1])
    rows = []
    for index in range(15):
        mask = ids == index
        if mask.any():
            # Nonempty bins are numbered consecutively, so empty ones don't leave gaps.
            rows.append(
                dict(bin=len(rows),
                     mean_uncertainty=float(u[mask].mean()),
                     mean_error=float(e[mask].mean()),
                     count=int(mask.sum())))
    return rows


def window_profile(concentration, error, uncertainty=None, bins=10):
    """Mean error and uncertainty in quantile bins of the sliding-window weight
    concentration (sampling.window_concentration), over the given pixels."""
    k = torch.from_numpy(np.ascontiguousarray(concentration, dtype=np.float64)).flatten()
    e = torch.from_numpy(np.ascontiguousarray(error, dtype=np.float64)).flatten()
    u = None if uncertainty is None else torch.from_numpy(
        np.ascontiguousarray(uncertainty, dtype=np.float64)).flatten()
    edges = torch.quantile(k, torch.linspace(0, 1, bins + 1, dtype=torch.float64))
    ids = torch.bucketize(k, edges[1:-1])
    rows = []
    for index in range(bins):
        mask = ids == index
        if mask.any():
            # Nonempty bins are numbered consecutively, as in _calibration_bins.
            rows.append(
                dict(bin=len(rows),
                     concentration_low=float(k[mask].min()),
                     concentration_high=float(k[mask].max()),
                     mean_error=float(e[mask].mean()),
                     mean_uncertainty=None if u is None else float(u[mask].mean()),
                     count=int(mask.sum())))
    return rows


def analyze(target,
            prediction,
            uncertainty,
            *,
            analysis='metrics',
            config=None,
            data_range=None,
            mask=None):
    """Evaluate one selected analysis, preserving its mask, reduction and normalization.

    mask, a boolean array over the spatial axes, restricts every built-in analysis to its
    pixels in place of the analyses' own implicit masks; custom analyses receive it as a
    keyword argument."""
    if target.shape != prediction.shape or target.ndim not in (3, 4):
        raise ValueError('Analysis requires matching C,H,W or C,D,H,W target and prediction')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Analysis inputs must be finite')
    if uncertainty is not None and (uncertainty.shape != target.shape
                                    or not np.isfinite(uncertainty).all() or
                                    (uncertainty < 0).any()):
        raise ValueError('Uncertainty must be finite, nonnegative, and match the target')
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target.shape[1:] or not mask.any():
            raise ValueError('mask must match the spatial shape and select at least one pixel')
    custom = config.inference.custom_analyses.get(analysis) if config else None
    if custom is not None:
        instance = import_object(custom.class_path)(**custom.kwargs)
        if instance.requires_uncertainty and uncertainty is None:
            raise ValueError(f'{analysis} requires uncertainty')
        extra = {} if mask is None else dict(mask=mask)
        return {analysis: instance(target, prediction, uncertainty, **extra)}
    if analysis != 'metrics' and uncertainty is None:
        raise ValueError('This analysis requires uncertainty')
    error = np.abs(prediction - target)

    def region(array):
        """Channel zero, restricted to the mask."""
        return array[0] if mask is None else array[0][mask]

    reference = module_for(config.framework).reference(config, analysis) if config else None
    if analysis == 'sparsification':
        if mask is not None:
            uncertainty, error = uncertainty[:, mask], error[:, mask]
        fractions, curve, random, oracle = sparsification(
            uncertainty, error, kind=reference.sparsification_kind if reference else 'fast')
        return {
            'sparsification': [
                dict(fraction=f, error=c, random_error=r, oracle_error=o)
                for f, c, r, o in zip(fractions, curve, random, oracle)
            ]
        }
    if analysis == 'calibration':
        return {'calibration': _tail_bins(region(uncertainty), region(error))}
    if analysis == 'uncertainty_summary':
        inside = target[0] != -1 if mask is None else mask
        return {'uncertainty_summary': [_summary(uncertainty[0][inside], error[0][inside])]}
    if analysis != 'metrics':
        raise ValueError(f'Unknown analysis: {analysis}')
    if reference and reference.metrics_kind == 'summary' and uncertainty is not None:
        return {'metrics': [_summary(region(uncertainty), region(error))]}
    gt, pred = region(target), region(prediction)
    # Without a mask or an uncertainty map, target pixels equal to zero are masked out and
    # the remaining foreground is flattened before computing MSE/PSNR/SSIM.
    if mask is None and uncertainty is None:
        foreground = gt != 0
        gt, pred = gt[foreground], pred[foreground]
    value_range = float(np.ptp(target[0] if mask is None else gt)) if data_range is None else data_range
    mse = float(np.mean((gt - pred)**2)) if gt.size else None
    metrics = dict(mse=mse, psnr=None, ssim=None)
    if gt.size:
        metrics['psnr'] = float('inf') if mse == 0 else (float(
            peak_signal_noise_ratio(gt, pred, data_range=value_range)) if value_range > 0 else None)
        # SSIM's default window is 7 pixels wide; a smaller or constant-intensity
        # crop leaves it undefined (left blank) rather than raising.
        if mask is None and min(gt.shape) >= 7 and value_range > 0:
            metrics['ssim'] = float(structural_similarity(gt, pred, data_range=value_range))
        elif mask is not None and min(target.shape[1:]) >= 7 and value_range > 0:
            # SSIM needs the image around each pixel: map it, then average over the mask.
            _, similarity = structural_similarity(target[0],
                                                  prediction[0],
                                                  data_range=value_range,
                                                  full=True)
            metrics['ssim'] = float(similarity[mask].mean())
    result = {'metrics': [metrics]}
    if uncertainty is not None:
        if mask is None:
            norm = norm_percentile(torch.from_numpy(uncertainty.copy())[None])[0, 0].numpy()
        else:
            norm = norm_percentile(torch.from_numpy(region(uncertainty).copy())[None])[0].numpy()
        metrics.update(_correlations(region(uncertainty), region(error)))
        metrics.update({
            f'{key}_norm': value
            for key, value in _correlations(norm, region(error)).items()
        })
        if reference and reference.calibration_bins:
            result['calibration_bins'] = _calibration_bins(region(uncertainty), region(error))
    return result


class Reporter:
    """Writes one analysis run's CSVs and predictions, with consistent lowercase headers
    and globally increasing sample IDs. Samples that come with regions (named boolean
    masks) are analyzed once per region, into one CSV per analysis and region, e.g.
    metrics_brain.csv."""

    def __init__(self, output_dir, config):
        self.root, self.config = Path(output_dir), config
        self.stack, self.writers, self.index = ExitStack(), {}, 0
        self.regions, self.concentration = [], {}

    def __enter__(self):
        names = list(self.config.inference.analyses)
        self.schemas = dict(SCHEMAS)
        if names:
            if len(names) != 1:
                raise ValueError('Each Reporter handles one analysis run')
            custom = self.config.inference.custom_analyses.get(names[0])
            if custom is not None:
                self.schemas[names[0]] = import_object(custom.class_path)(**custom.kwargs).columns
            elif module_for(self.config.framework).reference(self.config, names[0]).calibration_bins:
                names.append('calibration_bins')
            if self.config.inference.window_profile:
                names.append('window_profile')
        self.names = names
        if self.config.inference.save_predictions:
            (self.root / 'predictions').mkdir(exist_ok=True)
        return self

    def _writer(self, name, region):
        # Opened on first use: which regions exist is only known from the data.
        if (name, region) not in self.writers:
            stem = name if region is None else f'{name}_{region}'
            handle = self.stack.enter_context((self.root / f'{stem}.csv').open('w', newline=''))
            writer = csv.DictWriter(handle, fieldnames=['sample', 'case_id', *self.schemas[name]])
            writer.writeheader()
            self.writers[name, region] = writer
        return self.writers[name, region]

    def _analyze(self, target, prediction, uncertainty, mask):
        rows = analyze(target,
                       prediction,
                       uncertainty,
                       analysis=self.config.inference.analyses[0],
                       config=self.config,
                       data_range=self.config.inference.data_range,
                       mask=mask)
        if self.config.inference.window_profile:
            from .sampling import window_concentration
            shape = target.shape[1:]
            if shape not in self.concentration:
                self.concentration[shape] = window_concentration(shape, self.config)
            inside = (lambda array: array) if mask is None else (lambda array: array[mask])
            rows['window_profile'] = window_profile(
                inside(self.concentration[shape]), inside(np.abs(prediction - target)[0]),
                None if uncertainty is None else inside(uncertainty[0]))
        return rows

    def write(self, condition, target, result, case_ids=None, *, regions=None, affines=None):
        """regions: optional {name: N,1,*spatial boolean masks}; affines: optional N,4,4."""
        for i in range(len(condition)):
            c, p = condition[i].detach().cpu().numpy(), result.image[i].detach().cpu().numpy()
            t = None if target is None else target[i].detach().cpu().numpy()
            u = None if result.variance is None else result.variance[i].detach().cpu().numpy()
            case_id = str(case_ids[i]) if case_ids is not None else str(self.index)
            if self.names:
                if t is None:
                    raise ValueError(
                        'Error analyses require targets; use inference.analyses: [] for unlabeled data'
                    )
                masks = {None: None}
                if regions:
                    masks = {
                        name: torch.as_tensor(mask[i]).cpu().numpy().astype(bool).reshape(t.shape[1:])
                        for name, mask in regions.items()
                    }
                for region, mask in masks.items():
                    if mask is not None and not mask.any():
                        continue  # e.g. a subject without enhancing tumor.
                    if region is not None and region not in self.regions:
                        self.regions.append(region)
                    rows = self._analyze(t, p, u, mask)
                    for name in self.names:
                        for row in rows[name]:
                            self._writer(name, region).writerow(
                                dict(sample=self.index, case_id=case_id, **row))
            if self.config.inference.save_predictions:
                if self.config.inference.prediction_format == 'nifti':
                    self._save_nifti(case_id, p, u, None if affines is None else affines[i])
                else:
                    arrays = dict(condition=c, prediction=p, case_id=np.asarray(case_id))
                    if t is not None:
                        arrays['target'] = t
                    if u is not None:
                        arrays['variance'] = u
                    np.savez_compressed(self.root / 'predictions' / f'{self.index:06d}.npz',
                                        **arrays)
            self.index += 1

    def _save_nifti(self, case_id, prediction, variance, affine):
        """<case_id>_prediction.nii.gz and _variance.nii.gz, channels last when several."""
        if affine is None:
            raise ValueError("prediction_format='nifti' requires a dataset that returns an affine")
        import nibabel as nib
        affine = torch.as_tensor(affine).cpu().numpy().astype(np.float64)
        for name, array in (('prediction', prediction), ('variance', variance)):
            if array is not None:
                volume = array[0] if len(array) == 1 else np.moveaxis(array, 0, -1)
                nib.save(nib.Nifti1Image(volume.astype(np.float32), affine),
                         self.root / 'predictions' / f'{case_id}_{name}.nii.gz')

    def __exit__(self, *args):
        self.stack.close()
