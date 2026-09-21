"""ldm: DDIM sampling in a pretrained VAE's latent space, then decoded to pixels.

Mirrors test_LDM.py, test_LDM_aleatoric.py and test_LDM_aleatoric_two_forward.py.
"""
from .common import generate, select_reference
from dataclasses import replace

# Each name below is a label, not a callable: it is never imported or invoked. It
# records which of the three scripts above this combination of mode/uncertainty/
# analysis corresponds to, for traceability (run.json's legacy_function) and for the
# registry test that keeps this mapping from drifting. The actual computation is the
# shared, generic code in common.py, driven by the other fields on Reference.
REFERENCES = {
    'base': 'run_inference_LDM_vanilla_and_log',
    'posthoc': {
        'sparsification': 'run_inference_LDM_vanilla_and_log_MC_sampling_sparsification',
        'metrics': 'run_inference_LDM_vanilla_and_log_MC_sampling',
        'uncertainty_summary': 'run_inference_LDM_vanilla_and_log_MC_sampling_uncertainty_eval',
        'calibration':
        'run_inference_LDM_vanilla_and_log_MC_sampling_uncertainty_calibration_tail_bins'
    },
    'aleatoric': {
        'sparsification':
        'run_inference_LDM_aleatoric_and_log_uncertainty_propagation_sparsification',
        'metrics': 'run_inference_LDM_aleatoric_and_log_uncertainty_propagation'
    },
    'selfcond': {
        'sparsification':
        'run_inference_LDM_self_refining_and_log_uncertainty_propagation_sparsification',
        'metrics': 'run_inference_LDM_self_refining_and_log_uncertainty_propagation',
        'uncertainty_summary': 'run_inference_LDM_self_refining_and_log_uncertainty_eval',
        'calibration': 'run_inference_LDM_self_refining_and_log_uncertainty_calibration_tail_bins'
    },
    'ablation': 'run_inference_LDM_self_refining_and_log_uncertainty_propagation_ablation'
}


def reference(config, analysis="metrics"):
    last_k = 10
    flow_dt_squared = False
    selected = select_reference(config,
                                analysis,
                                REFERENCES,
                                last_k=last_k,
                                flow_dt_squared=flow_dt_squared,
                                context_autocast=True,
                                context_concat_allowed=True)
    return replace(selected, sparsification_kind="fast" if config.mode == "base" else "standard")


def infer(condition, models, config, *, analysis="metrics", propagate=None):
    """Run inference on already-scaled latent conditions (N,C,H,W)."""
    return generate(condition, models, config, reference(config, analysis), propagate=propagate)
