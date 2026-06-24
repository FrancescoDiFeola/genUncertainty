from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

PUBLIC_ANALYSES = (
    "metrics",
    "sparsification",
    "spatial_error_correlation",
    "calibration_bins",
)

ANALYSIS_ALIASES = {
    "metrics": "metrics",
    "metrics_no_uncertainty": "metrics",
    "sparsification": "sparsification",
    "uncertainty_eval": "spatial_error_correlation",
    "spatial_error_correlation": "spatial_error_correlation",
    "uncertainty_cal": "calibration_bins",
    "calibration_bins": "calibration_bins",
}

VALID_ANALYSES: Dict[str, Tuple[str, ...]] = {
    "base": ("metrics",),
    "aleatoric": ("metrics", "sparsification"),
    "selfcond": (
        "metrics",
        "sparsification",
        "spatial_error_correlation",
        "calibration_bins",
    ),
}

LEGACY_WRITER_TYPES = {
    "metrics": "metrics",
    "sparsification": "sparsification",
    "spatial_error_correlation": "uncertainty_eval",
    "calibration_bins": "uncertainty_cal",
}


def canonicalize_analysis_name(name: str) -> str:
    try:
        return ANALYSIS_ALIASES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(ANALYSIS_ALIASES))
        raise ValueError(f"Unknown analysis '{name}'. Valid analyses/aliases: {valid}") from exc


def canonicalize_analyses(analyses: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for name in analyses:
        canonical = canonicalize_analysis_name(str(name))
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


def validate_analyses(mode: str, analyses: Iterable[str]) -> List[str]:
    analyses = canonicalize_analyses(analyses)
    if not analyses:
        analyses = ["metrics"]
    valid = VALID_ANALYSES.get(mode)
    if valid is None:
        raise ValueError(f"Unknown inference mode '{mode}'.")
    invalid = [name for name in analyses if name not in valid]
    if invalid:
        raise ValueError(
            f"Analysis {invalid} not supported for mode='{mode}'. "
            f"Supported analyses: {list(valid)}."
        )
    return analyses
