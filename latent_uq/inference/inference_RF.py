"""fm: image-space rectified-flow sampling for base, aleatoric and self-conditioned models.

Mirrors test_RF.py, test_RF_aleatoric.py and test_RF_aleatoric_two_forward.py.
"""
from .common import generate, select_reference
from dataclasses import replace

# Each name below is a label, not a callable: it is never imported or invoked. It
# records which of the three scripts above this combination of mode/uncertainty/
# analysis corresponds to, for traceability (run.json's legacy_function) and for the
# registry test that keeps this mapping from drifting. The actual computation is the
# shared, generic code in common.py, driven by the other fields on Reference.
REFERENCES = {
    'base': 'run_inference_RF_vanilla',
    'posthoc': {
        'sparsification': 'run_inference_RF_vanilla_and_log_MC_sampling_sparsification',
        'metrics': 'run_inference_RF_vanilla_and_log_MC_sampling',
        'uncertainty_summary': 'run_inference_RF_vanilla_and_log_MC_sampling_uncertainty_eval',
        'calibration':
        'run_inference_RF_vanilla_and_log_MC_sampling_uncertainty_calibration_tail_bins'
    },
    'aleatoric': {
        'sparsification': 'run_inference_RF_aleatoric_sparsification',
        'metrics': 'run_inference_RF_aleatoric'
    },
    'selfcond': {
        'sparsification':
        'run_inference_RF_self_refining_and_log_v3_clean_unc_integral_sparsification',
        'metrics': 'run_inference_RF_self_refining_and_log_v3_clean_unc_integral',
        'uncertainty_summary': 'run_inference_and_log_v3_clean_uncertainty_eval',
        'calibration': 'run_inference_and_log_v3_clean_uncertainty_calibration_tail_bins'
    },
    'ablation': 'run_inference_RF_self_refining_and_log_v3_clean_unc_integral_ablation'
}


def reference(config, analysis="metrics"):
    last_k = 10 if config.mode == "selfcond" and analysis == "sparsification" else 30
    flow_dt_squared = config.mode == "selfcond" and analysis != "sparsification" and config.inference.self_conditioning
    selected = select_reference(config,
                                analysis,
                                REFERENCES,
                                last_k=last_k,
                                flow_dt_squared=flow_dt_squared)
    return replace(selected,
                   calibration_bins=analysis == "metrics" and config.mode != "base"
                   and config.uncertainty != "none",
                   sparsification_kind="standard" if config.mode == "aleatoric" else "fast")


def infer(condition, models, config, *, analysis="metrics", propagate=None):
    """Run inference on image-space conditions (N,C,H,W)."""
    return generate(condition, models, config, reference(config, analysis), propagate=propagate)
