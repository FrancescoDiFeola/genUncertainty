"""Numerical parity with the actual functions and trainer bodies in the selected ZIP."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from latent_uq.inference import module_for
from latent_uq.process import Process
from latent_uq.training import train_step
from latent_uq.analysis import analyze
from tests.reference_cases import setup, seed, images

ROOT = Path(__file__).with_name('fixtures')
MANIFEST = json.loads((ROOT / 'legacy_reference.json').read_text())
CASES = MANIFEST['cases']

# train_LDM_aleatoric_two_forward.py resampled noise/timesteps before its second forward
# and excluded the context encoder from the optimizer, unlike the other three retained
# trainers. latent_uq deliberately unifies LDM selfcond training with dm/fm/lfm instead of
# reproducing that divergence, so these two fixture cases no longer apply.
UNIFIED_SELFCOND_TRAINING_DEVIATIONS = {'train_ldm_selfcond', 'train_ldm_selfcond_prediction_variance'}


def _case_param(case):
    if case['id'] in UNIFIED_SELFCOND_TRAINING_DEVIATIONS:
        return pytest.param(
            case,
            marks=pytest.mark.skip(
                reason='LDM selfcond training is intentionally unified with dm/fm/lfm; '
                'no longer reproduces train_LDM_aleatoric_two_forward.py resampling/frozen-encoder behavior'
            ))
    return case


@pytest.mark.parametrize('case', [_case_param(c) for c in CASES], ids=lambda c: c['id'])
def test_selected_legacy_implementation(case):
    options = {
        k: case[k]
        for k in ('analysis', 'posthoc', 'ablation', 'calibration', 'concat') if k in case
    }
    config, models = setup(case['framework'], case['mode'], **options)
    condition, target = images()
    with np.load(ROOT / 'legacy_reference.npz') as expected:
        if case['kind'] == 'inference':
            module = module_for(config.framework)
            ref = module.reference(config, case['analysis'])
            assert ref.function == case['function']
            encoded = models.vae.encode(
                condition)[0] * config.model.scaling_factor if models.vae else condition
            seed()
            with torch.no_grad():
                actual = module.infer(encoded,
                                      models,
                                      config,
                                      analysis=case['analysis'],
                                      propagate=config.uncertainty == 'propagated')
            for suffix, value in [('mean', actual.mean), ('variance', actual.variance)]:
                key = case['id'] + '__' + suffix
                if key not in expected:
                    assert value is None
                else:
                    torch.testing.assert_close(value,
                                               torch.from_numpy(expected[key]),
                                               rtol=2e-6,
                                               atol=2e-6)
            rows = analyze(target[0].numpy(),
                           actual.image[0].numpy(),
                           None if actual.variance is None else actual.variance[0].numpy(),
                           analysis=case['analysis'],
                           config=config)
            compare_rows(rows[case['analysis']], case['analysis_rows'])
            compare_rows(rows.get('calibration_bins', []), case['calibration_rows'])
        else:
            parameters = [
                p for module in models.modules() if module is not None for p in module.parameters()
                if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(parameters,
                                          lr=config.training.lr,
                                          weight_decay=config.training.weight_decay)
            seed()
            loss = train_step(condition, target, models, Process(config), config)
            torch.testing.assert_close(loss.detach(),
                                       torch.from_numpy(expected[case['id'] + '__loss']),
                                       rtol=2e-6,
                                       atol=2e-6)
            loss.backward()
            for part in ('backbone', 'context', 'uncertainty_decoder'):
                module = getattr(models, part)
                if module is None:
                    continue
                for name, p in module.named_parameters():
                    key = f"{case['id']}__grad__{part}__{name}"
                    if key in expected:
                        torch.testing.assert_close(p.grad,
                                                   torch.from_numpy(expected[key]),
                                                   rtol=3e-6,
                                                   atol=2e-6)
            optimizer.step()
            for part in ('backbone', 'context', 'uncertainty_decoder'):
                module = getattr(models, part)
                if module is None:
                    continue
                for name, p in module.named_parameters():
                    key = f"{case['id']}__weight__{part}__{name}"
                    torch.testing.assert_close(p.detach(),
                                               torch.from_numpy(expected[key]),
                                               rtol=2e-6,
                                               atol=2e-6)
        assert len(models.backbone.calls) == case['model_calls']


def compare_rows(actual, reference):
    """Column names/IDs are normalized; numeric definitions must match the original rows."""
    names = {
        'MSE': 'mse',
        'MAE': 'mae',
        'PSNR': 'psnr',
        'SSIM': 'ssim',
        'top5_u_mean': 'u_top5_mean',
        'Fraction': 'fraction',
        'Error': 'error',
        'RandomError': 'random_error',
        'OracleError': 'oracle_error',
        'Bin': 'bin',
        'Unc_mean': 'mean_uncertainty',
        'Err_mean': 'mean_error',
        'Count': 'count'
    }
    assert len(actual) == len(reference)
    for observed, original in zip(actual, reference):
        expected = {}
        for key, value in original.items():
            if key in {'Sample', 'sample', 'Type'}:
                continue
            key = names.get(key, key.replace('_u_unnorm', '').replace('_u_norm', '_norm').lower())
            expected[key] = value
        assert observed.keys() == expected.keys()
        for key, value in expected.items():
            if value is None or not np.isfinite(value):
                assert observed[key] is None or not np.isfinite(observed[key])
            else:
                assert observed[key] == pytest.approx(value, rel=6e-6, abs=3e-6), key


def test_reference_registry_contains_exactly_the_retained_test_calls():
    for framework in ('dm', 'fm', 'ldm', 'lfm'):
        names = module_for(framework).REFERENCES
        registered = {
            name
            for item in names.values()
            for name in ([item] if isinstance(item, str) else item.values())
        }
        expected = {
            c['function']
            for c in CASES if c['kind'] == 'inference' and c['framework'] == framework
        }
        assert registered == expected
