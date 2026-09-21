"""Rebuild reference fixtures from the user-selected ZIP; never import its entrypoints.

Usage: python tests/build_legacy_reference.py /path/to/genUncertainty-main.zip
The resulting numerical fixtures let ordinary tests run without that ZIP.
"""
import ast
import copy
import hashlib
import inspect
import json
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.networks.schedulers import RFlowScheduler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tests.reference_cases import setup, seed, images


def main(archive_path):
    torch.set_num_threads(1)
    with zipfile.ZipFile(archive_path) as archive:
        sources = {
            name.split('/', 1)[1]: archive.read(name).decode('utf-8-sig')
            for name in archive.namelist()
            if name.startswith('genUncertainty-main/') and name.endswith('.py')
        }
    output, cases = {}, []
    shared = dict(torch=torch,
                  np=np,
                  F=F,
                  autocast=lambda *a, **k: nullcontext(),
                  tqdm=lambda x, **k: x,
                  print=lambda *a, **k: None,
                  Callable=Callable,
                  partial=partial,
                  SPADEDiffusionModelUNet=type('UnusedSPADE', (), {}),
                  compute_psnr=peak_signal_noise_ratio,
                  compute_ssim=structural_similarity,
                  pearsonr=pearsonr,
                  spearmanr=spearmanr,
                  roc_auc_score=roc_auc_score)

    class Plot:

        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

        def subplots(self, nrows=1, ncols=1, **kwargs):
            axes = np.empty((nrows, ncols), dtype=object)
            for index in np.ndindex(axes.shape):
                axes[index] = Plot()
            return Plot(), axes.squeeze()

    class Recorder:

        def __init__(self):
            self.rows = []

        def writerow(self, row):
            self.rows.append({
                k: float(v) if isinstance(v, (np.number, torch.Tensor)) else v
                for k, v in row.items()
            })

    shared['plt'] = Plot()

    def function(path, name):
        node = copy.deepcopy(
            next(n for n in ast.parse(sources[path]).body
                 if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []
        if name.startswith('run_'):
            node.body.append(ast.Return(ast.Call(ast.Name('locals', ast.Load()), [], [])))
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        scope = dict(shared)
        exec(compile(tree, path, 'exec'), scope)
        return scope[name]

    shared['norm_percentile'] = function('src/inference/utils.py', 'norm_percentile')
    for name in ('sparsification_curve_fast', 'random_sparsification_fast', 'sparsification_curve',
                 'random_sparsification', 'uncertainty_error_tail_bins_torch',
                 'collect_calibration_data', 'summarize_uncertainty',
                 'map_correlations_multi_thresholds', 'map_correlations'):
        shared[name] = function('src/inference/utils.py', name)
    inferer_class = next(n for n in ast.parse(sources['inferers/inferer.py']).body
                         if isinstance(n, ast.ClassDef) and n.name == 'DiffusionInferer')
    inferer_method = copy.deepcopy(
        next(n for n in inferer_class.body
             if isinstance(n, ast.FunctionDef) and n.name == '__call__'))
    inferer_method.decorator_list = []
    scope = dict(shared)
    exec(compile(ast.Module(body=[inferer_method], type_ignores=[]), 'inferers/inferer.py', 'exec'),
         scope)
    inferer_call = scope['__call__']

    def scheduler(framework, mode, training=False):
        if framework in {'dm', 'ldm'}:
            cls = DDPMScheduler if training else DDIMScheduler
            s = cls(num_train_timesteps=1000,
                    beta_start=0.0015,
                    beta_end=0.0205,
                    schedule='scaled_linear_beta',
                    **({} if training else dict(clip_sample=False)))
            if not training:
                s.set_timesteps(50)
            return s
        base = 64**2 if framework == 'lfm' else 256**2
        if not training and framework == 'lfm' and mode == 'aleatoric':
            base = 256**2
        s = RFlowScheduler(num_train_timesteps=1000,
                           use_discrete_timesteps=False,
                           sample_method='uniform',
                           use_timestep_transform=True,
                           base_img_size_numel=base,
                           spatial_dim=2)
        if not training:
            s.set_timesteps(30, input_img_size_numel=base)
        return s

    script_sets = {
        'dm':
        ('ddpm', 'test_ddpm.py', 'test_ddpm_aleatoric.py', 'test_ddpm_aleatoric_two_forward.py'),
        'fm': ('RF', 'test_RF.py', 'test_RF_aleatoric.py', 'test_RF_aleatoric_two_forward.py'),
        'ldm': ('LDM', 'test_LDM.py', 'test_LDM_aleatoric.py', 'test_LDM_aleatoric_two_forward.py'),
        'lfm': ('LFM', 'test_LFM.py', 'test_LFM_aleatoric.py', 'test_LFM_two_forward.py')
    }
    for framework, (family, *scripts) in script_sets.items():
        for mode, script in zip(['base', 'aleatoric', 'selfcond'], scripts):
            calls = sorted([
                n for n in ast.walk(ast.parse(sources[script])) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id.startswith('run_')
            ],
                           key=lambda n: n.lineno)
            for call in calls:
                for concat in ([False, True] if
                               (framework, mode) == ('ldm', 'selfcond') else [False]):
                    name = call.func.id
                    analysis = ('sparsification' if 'sparsification' in name else
                                'calibration' if 'calibration' in name else
                                'uncertainty_summary' if 'uncertainty_eval' in name else 'metrics')
                    posthoc, ablation = 'MC_sampling' in name, 'ablation' in name
                    config, models = setup(framework,
                                           mode,
                                           analysis=analysis,
                                           posthoc=posthoc,
                                           ablation=ablation,
                                           concat=concat)
                    condition, target = images()
                    encoded = models.vae.encode(
                        condition)[0] * config.model.scaling_factor if models.vae else condition
                    path = f'src/inference/inference_{family}.py'
                    fn = function(path, name)
                    primary, secondary = Recorder(), Recorder()
                    k = 10 if framework in ('dm',
                                            'ldm') or (framework == 'fm' and mode == 'selfcond'
                                                       and analysis == 'sparsification') else 30
                    values = dict(diffusion_model=models.backbone,
                                  context_encoder=models.context,
                                  autoencoder=models.vae,
                                  channels=2 if concat else 3 if config.latent else 1,
                                  condition_batch=encoded,
                                  gt_batch=target,
                                  writer=Plot(),
                                  step=0,
                                  device='cpu',
                                  scheduler=scheduler(framework, mode),
                                  dir='unused',
                                  scaling=config.model.scaling_factor,
                                  csv_writer=primary,
                                  csv_writer_2=secondary,
                                  K=k,
                                  k_steps=10,
                                  n_sampling=4)
                    kwargs = {
                        n: values[n]
                        for n, p in inspect.signature(fn).parameters.items() if n in values
                    }
                    seed()
                    with torch.no_grad():
                        result = fn(**kwargs)
                    ident = f'{framework}_{mode}_{analysis}' + ('_posthoc' if posthoc else '') + (
                        '_ablation' if ablation else '') + ('_prediction_variance'
                                                            if concat else '')
                    output[ident + '__mean'] = next(result[n]
                                                    for n in ('pred_denoised', 'final_output')
                                                    if n in result).numpy()
                    variance = result.get('mc_uncertainty_map') if posthoc else result.get(
                        'uncertainty_map') if mode != 'base' else None
                    if variance is not None:
                        output[ident + '__variance'] = variance.numpy()
                    cases.append(
                        dict(id=ident,
                             kind='inference',
                             framework=framework,
                             mode=mode,
                             analysis=analysis,
                             posthoc=posthoc,
                             ablation=ablation,
                             concat=concat,
                             script=script,
                             call_line=call.lineno,
                             function=name,
                             source=path,
                             analysis_rows=primary.rows,
                             calibration_rows=secondary.rows,
                             model_calls=len(models.backbone.calls)))
        # Extract the actual per-batch trainer body through the loss, before backward/logging.
        for mode, suffix in [('base', ''), ('aleatoric', '_aleatoric'),
                             ('selfcond', '_aleatoric_two_forward')]:
            path = f'train_{family}{suffix}.py'
            loops = [
                n for n in ast.walk(ast.parse(sources[path])) if isinstance(n, ast.For) and any(
                    isinstance(x, ast.Name) and x.id == 'batch' for x in ast.walk(n.target))
            ]
            body = None
            for loop in loops:
                cut = next((
                    i for i, n in enumerate(loop.body)
                    if isinstance(n, ast.Expr) and 'scaler.scale(loss).backward' in ast.unparse(n)),
                           None)
                if cut is not None:
                    body = copy.deepcopy(loop.body[:cut])
                    break
            assert body is not None, path
            variants = [(False, False)]
            if (framework, mode) == ('ldm', 'aleatoric'):
                variants.append((True, False))
            if mode == 'selfcond':
                variants.append((False, True))
            for calibration, concat in variants:
                config, models = setup(framework, mode, calibration=calibration, concat=concat)
                params = list(models.backbone.parameters())
                if models.context and framework != 'ldm':
                    params += list(models.context.parameters())
                if models.uncertainty_decoder:
                    params += list(models.uncertainty_decoder.parameters())
                optimizer = torch.optim.AdamW(params, lr=1.5e-5)
                sched = scheduler(framework, mode, training=True)
                instance = SimpleNamespace(scheduler=sched)
                inferer = lambda **kwargs: inferer_call(instance, **kwargs)
                condition, target = images()
                local = dict(shared,
                             DEVICE='cpu',
                             batch={
                                 'A': condition,
                                 'B': target
                             },
                             diffusion=models.backbone,
                             autoencoder=models.vae,
                             spatial_encoder=models.context,
                             scheduler=sched,
                             optimizer=optimizer,
                             inferer=inferer,
                             scaling_factor=config.model.scaling_factor,
                             uncertainty_decoder=models.uncertainty_decoder,
                             args=SimpleNamespace(
                                 batch_size=1,
                                 diff_loss_weight=1.0,
                                 spatial_enc_channels=2 if concat else 3 if config.latent else 1,
                                 uncertainty_calibration=calibration,
                                 uncertainty_loss_weight=0.01))
                if mode != 'base':
                    local['heteroscedastic_loss'] = function(path, 'heteroscedastic_loss')
                if calibration:
                    local['uncertainty_calibration_loss'] = function(
                        path, 'uncertainty_calibration_loss')
                seed()
                exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), local)
                loss = local['loss']
                loss.backward()
                ident = f'train_{framework}_{mode}' + ('_calibration' if calibration else '') + (
                    '_prediction_variance' if concat else '')
                output[ident + '__loss'] = loss.detach().numpy()
                for part in ('backbone', 'context', 'uncertainty_decoder'):
                    module = getattr(models, part)
                    if module is not None:
                        for n, p in module.named_parameters():
                            if p.grad is not None and not (part == 'context'
                                                           and framework == 'ldm'):
                                output[f'{ident}__grad__{part}__{n}'] = p.grad.numpy().copy()
                optimizer.step()
                for part in ('backbone', 'context', 'uncertainty_decoder'):
                    module = getattr(models, part)
                    if module is not None:
                        for n, p in module.named_parameters():
                            output[f'{ident}__weight__{part}__{n}'] = p.detach().numpy().copy()
                cases.append(
                    dict(id=ident,
                         kind='training',
                         framework=framework,
                         mode=mode,
                         calibration=calibration,
                         concat=concat,
                         source=path,
                         body_line=body[0].lineno,
                         model_calls=len(models.backbone.calls)))
    folder = Path(__file__).with_name('fixtures')
    np.savez_compressed(folder / 'legacy_reference.npz', **output)
    manifest = dict(archive_sha256=hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
                    torch=str(torch.__version__),
                    monai=__import__('monai').__version__,
                    seed=29,
                    cases=cases,
                    source_hashes={
                        p: hashlib.sha256(sources[p].encode()).hexdigest()
                        for p in sorted({c['source']
                                         for c in cases}
                                        | {c['script']
                                           for c in cases if 'script' in c}
                                        | {'inferers/inferer.py', 'src/inference/utils.py'})
                    })
    (folder / 'legacy_reference.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Generated {len(cases)} independent reference cases, {len(output)} arrays')


if __name__ == '__main__':
    main(sys.argv[1])
