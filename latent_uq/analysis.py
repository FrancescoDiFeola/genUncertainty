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
        """target, prediction: (C,H,W) numpy arrays. uncertainty: same shape or None.

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


def analyze(target, prediction, uncertainty, *, analysis='metrics', config=None, data_range=None):
    """Evaluate one selected analysis, preserving its mask, reduction and normalization."""
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError('Analysis requires matching C,H,W target and prediction')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Analysis inputs must be finite')
    if uncertainty is not None and (uncertainty.shape != target.shape
                                    or not np.isfinite(uncertainty).all() or
                                    (uncertainty < 0).any()):
        raise ValueError('Uncertainty must be finite, nonnegative, and match the target')
    custom = config.inference.custom_analyses.get(analysis) if config else None
    if custom is not None:
        instance = import_object(custom.class_path)(**custom.kwargs)
        if instance.requires_uncertainty and uncertainty is None:
            raise ValueError(f'{analysis} requires uncertainty')
        return {analysis: instance(target, prediction, uncertainty)}
    if analysis != 'metrics' and uncertainty is None:
        raise ValueError('This analysis requires uncertainty')
    error = np.abs(prediction - target)
    reference = module_for(config.framework).reference(config, analysis) if config else None
    if analysis == 'sparsification':
        fractions, curve, random, oracle = sparsification(
            uncertainty, error, kind=reference.sparsification_kind if reference else 'fast')
        return {
            'sparsification': [
                dict(fraction=f, error=c, random_error=r, oracle_error=o)
                for f, c, r, o in zip(fractions, curve, random, oracle)
            ]
        }
    if analysis == 'calibration':
        return {'calibration': _tail_bins(uncertainty[0], error[0])}
    if analysis == 'uncertainty_summary':
        mask = target[0] != -1
        return {'uncertainty_summary': [_summary(uncertainty[0][mask], error[0][mask])]}
    if analysis != 'metrics':
        raise ValueError(f'Unknown analysis: {analysis}')
    if reference and reference.metrics_kind == 'summary' and uncertainty is not None:
        return {'metrics': [_summary(uncertainty[0], error[0])]}
    gt, pred = target[0], prediction[0]
    # Without an uncertainty map, target pixels equal to zero are masked out and the
    # remaining foreground is flattened before computing MSE/PSNR/SSIM.
    if uncertainty is None:
        mask = gt != 0
        gt, pred = gt[mask], pred[mask]
    value_range = float(np.ptp(target[0])) if data_range is None else data_range
    mse = float(np.mean((gt - pred)**2)) if gt.size else None
    metrics = dict(mse=mse, psnr=None, ssim=None)
    if gt.size:
        metrics['psnr'] = float('inf') if mse == 0 else (float(
            peak_signal_noise_ratio(gt, pred, data_range=value_range)) if value_range > 0 else None)
        # SSIM's default window is 7 pixels wide; a smaller or constant-intensity
        # crop leaves it undefined (left blank) rather than raising.
        if min(gt.shape) >= 7 and value_range > 0:
            metrics['ssim'] = float(structural_similarity(gt, pred, data_range=value_range))
    result = {'metrics': [metrics]}
    if uncertainty is not None:
        norm = norm_percentile(torch.from_numpy(uncertainty.copy())[None])[0, 0].numpy()
        metrics.update(_correlations(uncertainty[0], error[0]))
        metrics.update({
            f'{key}_norm': value
            for key, value in _correlations(norm, error[0]).items()
        })
        if reference and reference.calibration_bins:
            result['calibration_bins'] = _calibration_bins(uncertainty[0], error[0])
    return result


class Reporter:
    """Writes one analysis run's CSV and predictions, with consistent lowercase
    headers and globally increasing sample IDs."""

    def __init__(self, output_dir, config):
        self.root, self.config = Path(output_dir), config
        self.stack, self.writers, self.index = ExitStack(), {}, 0

    def __enter__(self):
        names = list(self.config.inference.analyses)
        schemas = dict(SCHEMAS)
        if names:
            if len(names) != 1:
                raise ValueError('Each Reporter handles one analysis run')
            custom = self.config.inference.custom_analyses.get(names[0])
            if custom is not None:
                schemas[names[0]] = import_object(custom.class_path)(**custom.kwargs).columns
            elif module_for(self.config.framework).reference(self.config, names[0]).calibration_bins:
                names.append('calibration_bins')
        try:
            for name in names:
                handle = self.stack.enter_context((self.root / f'{name}.csv').open('w', newline=''))
                writer = csv.DictWriter(handle, fieldnames=['sample', 'case_id', *schemas[name]])
                writer.writeheader()
                self.writers[name] = writer
            if self.config.inference.save_predictions:
                (self.root / 'predictions').mkdir(exist_ok=True)
            return self
        except Exception:
            self.stack.close()
            raise

    def write(self, condition, target, result, case_ids=None):
        for i in range(len(condition)):
            c, p = condition[i].detach().cpu().numpy(), result.image[i].detach().cpu().numpy()
            t = None if target is None else target[i].detach().cpu().numpy()
            u = None if result.variance is None else result.variance[i].detach().cpu().numpy()
            case_id = str(case_ids[i]) if case_ids is not None else str(self.index)
            if self.writers:
                if t is None:
                    raise ValueError(
                        'Error analyses require targets; use inference.analyses: [] for unlabeled data'
                    )
                rows = analyze(t,
                               p,
                               u,
                               analysis=self.config.inference.analyses[0],
                               config=self.config,
                               data_range=self.config.inference.data_range)
                for name, writer in self.writers.items():
                    for row in rows[name]:
                        writer.writerow(dict(sample=self.index, case_id=case_id, **row))
            if self.config.inference.save_predictions:
                arrays = dict(condition=c, prediction=p, case_id=np.asarray(case_id))
                if t is not None:
                    arrays['target'] = t
                if u is not None:
                    arrays['variance'] = u
                np.savez_compressed(self.root / 'predictions' / f'{self.index:06d}.npz', **arrays)
            self.index += 1

    def __exit__(self, *args):
        self.stack.close()
