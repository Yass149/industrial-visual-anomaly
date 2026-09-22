"""A small drift warning, not a diagnosis or an automatic retraining trigger."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Compare image-level feature projections in disjoint incoming windows to training.

    Patches within an image are correlated: treating each patch as an independent sample
    would manufacture statistical confidence. Instead each image supplies one mean vector.
    """

    def __init__(
        self,
        reference: np.ndarray,
        window_size: int = 64,
        alpha: float = 0.01,
        projections: int = 8,
        seed: int = 42,
    ) -> None:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or len(reference) < 2 or not np.isfinite(reference).all():
            raise ValueError("drift reference requires at least two finite image vectors")
        if window_size < 2 or projections < 1 or not 0 < alpha < 1:
            raise ValueError("invalid drift settings")
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(size=(reference.shape[1], projections))
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.reference = reference @ self.projection
        self.window_size, self.alpha = window_size, alpha
        self.pending: list[np.ndarray] = []
        self.last_check: dict[str, object] | None = None

    def update(self, feature: np.ndarray) -> dict[str, object]:
        """Append one image; test only complete nonoverlapping windows and log warnings."""
        feature = np.asarray(feature)
        if feature.shape != (self.projection.shape[0],) or not np.isfinite(feature).all():
            raise ValueError("incoming feature dimension or values are invalid")
        self.pending.append(feature @ self.projection)
        checked = len(self.pending) == self.window_size
        if checked:
            incoming = np.stack(self.pending)
            p_values = [
                float(ks_2samp(self.reference[:, j], incoming[:, j]).pvalue)
                for j in range(self.reference.shape[1])
            ]
            # Bonferroni controls testing across projections within one window, not over time.
            adjusted = min(1.0, min(p_values) * len(p_values))
            self.last_check = {
                "drift_detected": adjusted < self.alpha,
                "adjusted_p_value": adjusted,
                "window_images": len(incoming),
            }
            self.pending.clear()
            if self.last_check["drift_detected"]:
                logger.warning("feature drift detected: %s", self.last_check)
        return {
            "checked": checked,
            "pending_images": len(self.pending),
            "window_size": self.window_size,
            "last_check": self.last_check,
        }
