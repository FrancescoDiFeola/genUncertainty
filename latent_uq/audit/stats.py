"""Statistics for the audit: cluster bootstrap over subjects and critical sizes.

Lesions of the same subject are not independent, so every confidence interval resamples
subjects, not lesions. The mixed-effects logistic model of the plan (detection on
log-diameter and contrast, random effect per subject) needs statsmodels or R's lme4,
neither installed yet; the bootstrap estimates below do not need them.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

DIAMETER_BINS_MM = (0.0, 3.0, 5.0, 10.0, 20.0, np.inf)
BOOTSTRAP_SAMPLES = 2000


def cluster_bootstrap(rows: Sequence[dict[str, Any]], statistic: Callable[[list], float],
                      group: str = "case_id", samples: int = BOOTSTRAP_SAMPLES,
                      alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, low, high): `statistic` on all rows, and its percentile interval from
    resampling groups with replacement."""
    raise NotImplementedError


def binned_summary(rows: Sequence[dict[str, Any]], size: str, value: str,
                   bins: Sequence[float] = DIAMETER_BINS_MM) -> list[dict[str, Any]]:
    """Per size bin: count, median of `value` and its cluster-bootstrap interval."""
    raise NotImplementedError


def critical_size(rows: Sequence[dict[str, Any]], size: str = "diameter_mm",
                  detected: str = "detected_candidate", probability: float = 0.5
                  ) -> tuple[float, float, float]:
    """Size at which the detection probability reaches `probability` (d50, d90 with 0.9),
    from a logistic fit on log(size), with a cluster-bootstrap interval."""
    raise NotImplementedError
