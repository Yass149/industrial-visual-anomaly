"""Choose an operating point using calibration labels and explicit business assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    expected_cost: float
    false_positive_rate: float
    false_negative_rate: float
    cost_ratio: float
    base_rate: float

    def to_dict(self) -> dict[str, float]:
        """JSON-safe record, with costs measured in false-alarm cost units per image."""
        return asdict(self)


def validate_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require both classes so FPR and FNR are defined rather than silently invented."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("labels and scores must be matching finite one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("both binary classes (0 and 1) are required")
    return labels, scores


def optimise_threshold(
    labels: np.ndarray, scores: np.ndarray, cost_ratio: float, base_rate: float
) -> OperatingPoint:
    """Minimise C_FN * prevalence * FNR + (1-prevalence) * FPR on calibration data.

    A score >= threshold is an alarm. Include both alarm-all and alarm-none; on exact
    cost ties choose the larger threshold (fewer false alarms). Scores are not probabilities.
    """
    labels, scores = validate_scores(labels, scores)
    if not np.isfinite(cost_ratio) or cost_ratio <= 0 or not 0 < base_rate < 1:
        raise ValueError("cost_ratio must be finite and positive; base_rate must be in (0, 1)")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    # sklearn's +inf sentinel is not valid JSON; nextafter represents the same calibration rule.
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    if not np.isfinite(thresholds[0]):
        raise ValueError("scores too large to represent the alarm-none threshold")
    costs = cost_ratio * base_rate * (1 - tpr) + (1 - base_rate) * fpr
    best = int(np.argmin(costs))
    return OperatingPoint(
        float(thresholds[best]),
        float(costs[best]),
        float(fpr[best]),
        float(1 - tpr[best]),
        float(cost_ratio),
        float(base_rate),
    )
