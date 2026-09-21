"""dm: image-space DDIM sampling for base, aleatoric and self-conditioned models.
Mirrors test_ddpm.py, test_ddpm_aleatoric.py and test_ddpm_aleatoric_two_forward.py.
"""
from .common import generate, select_reference
from dataclasses import replace

# Each name below is a label, not a callable: it is never imported or invoked. It
# records which of the three scripts above this combination of mode/uncertainty/
# analysis corresponds to, for traceability (run.json's legacy_function) and for the
# registry test that keeps this mapping from drifting. The actual computation is the
# shared, generic code in common.py, driven by the other fields on Reference.
REFERENCES = {
    'base': 'run_ddpm_vanilla_inference_and_log',
    'posthoc': {
        'sparsification':
        'run_ddpm_vanilla_inference_and_log_MC_sampling_sparsification',
        'metrics':
        'run_ddpm_vanilla_inference_and_log_MC_sampling',
        'uncertainty_summary':
        'run_ddpm_vanilla_inference_and_log_MC_sampling_uncertainty_eval',
        'calibration':
        'run_ddpm_vanilla_inference_and_log_MC_sampling_uncertainty_calibration_tail_bins'
    },
    'aleatoric': {
        'sparsification': 'run_ddpm_aleatoric_inference_and_log_v2_sparsification',
        'metrics': 'run_ddpm_aleatoric_inference_and_log_v2'
    },
    'selfcond': {
        'sparsification': 'run_inference_and_log_v3_clean_unc_integral_sparsification',
        'metrics': 'run_inference_and_log_v3_clean_unc_integral',
        'uncertainty_summary': 'run_inference_and_log_v3_clean_uncertainty_eval',
        'calibration': 'run_inference_and_log_v3_clean_uncertainty_calibration_tail_bins'
    },
    'ablation': 'run_inference_and_log_v3_clean_unc_integral_ablation'
}


def reference(config, analysis="metrics"):
    last_k = 10
    flow_dt_squared = False
    selected = select_reference(config,
                                analysis,
                                REFERENCES,
                                last_k=last_k,
                                flow_dt_squared=flow_dt_squared)
    # dm/selfcond's metrics analysis reports MAE and the uncertainty-summary
    # statistics instead of MSE/PSNR/SSIM (the ablation still uses the latter).
    summary = config.mode == "selfcond" and config.inference.self_conditioning
    if analysis == "metrics":
        selected = replace(selected,
                           metrics_kind="summary" if summary else "reconstruction",
                           calibration_bins=config.uncertainty != "none" and not summary)
    return selected


def infer(condition, models, config, *, analysis="metrics", propagate=None):
    """Run inference on image-space conditions (N,C,H,W)."""
    return generate(condition, models, config, reference(config, analysis), propagate=propagate)
