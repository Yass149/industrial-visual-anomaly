"""Leakage-aware image metrics and reproducible calibration/evaluation partitioning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from anomaly.threshold import validate_scores


def split_indices(
    defect_types: list[str], fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify by defect type, preserving at least one example per side.

    MVTec supplies no labelled validation split. This portfolio protocol sacrifices part
    of its test set for calibration; the remaining metrics are not official full-test scores.
    """
    if not 0 < fraction < 1:
        raise ValueError("calibration fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    calibration, evaluation = [], []
    names = np.asarray(defect_types)
    for name in sorted(set(defect_types)):
        indices = np.flatnonzero(names == name)
        if len(indices) < 2:
            raise ValueError(f"need at least two examples of {name!r} to split honestly")
        indices = rng.permutation(indices)
        count = max(1, min(len(indices) - 1, int(round(len(indices) * fraction))))
        calibration.extend(indices[:count])
        evaluation.extend(indices[count:])
    return np.sort(calibration), np.sort(evaluation)


def image_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cost_ratio: float,
    base_rate: float,
) -> dict[str, object]:
    """Return raw benchmark PR and a prevalence-adjusted deployment scenario separately."""
    labels, scores = validate_scores(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    fpr, fnr = fp / (tn + fp), fn / (tp + fn)
    # Importance weighting changes only the class prior, assuming class-conditional stability.
    observed_rate = labels.mean()
    weights = np.where(
        labels == 1, base_rate / observed_rate, (1 - base_rate) / (1 - observed_rate)
    )
    adjusted_precision, adjusted_recall, _ = precision_recall_curve(
        labels,
        scores,
        sample_weight=weights,
    )
    denominator = base_rate * (1 - fnr) + (1 - base_rate) * fpr
    return {
        "n_images": len(labels),
        "n_normal": int(tn + fp),
        "n_anomalous": int(tp + fn),
        "observed_defect_rate": float(observed_rate),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "scenario_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(1 - fnr),
        "false_positive_rate": float(fpr),
        "scenario_precision": float(base_rate * (1 - fnr) / denominator) if denominator else None,
        "scenario_expected_cost": float(cost_ratio * base_rate * fnr + (1 - base_rate) * fpr),
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "scenario_pr_curve": {
            "precision": adjusted_precision.tolist(),
            "recall": adjusted_recall.tolist(),
        },
    }
