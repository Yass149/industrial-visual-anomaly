from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from anomaly.config import load_config
from anomaly.patchcore import PatchCore


class TinyExtractor(nn.Module):
    """Deterministic spatial features; exercise real scoring without a network download."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(images, (4, 4))


@pytest.fixture
def cfg(tmp_path):
    original = load_config()
    return original.model_copy(
        update={
            "model": original.model.model_copy(
                update={
                    "image_size": 32,
                    "crop_size": 32,
                    "gaussian_blur_sigma": 0.0,
                    "coreset_ratio": 0.5,
                }
            ),
            "serve": original.serve.model_copy(
                update={"artifacts_dir": tmp_path, "max_upload_mb": 1}
            ),
            "eval": original.eval.model_copy(update={"results_dir": tmp_path / "results"}),
        }
    )


@pytest.fixture
def tiny_model(monkeypatch, cfg):
    monkeypatch.setattr("anomaly.patchcore.PatchFeatureExtractor", lambda **kwargs: TinyExtractor())
    model = PatchCore(cfg)
    # Smooth normal intensity variations; the anomalous fixture inserts a local bright patch.
    generator = torch.Generator().manual_seed(12)
    images = torch.rand(12, 3, 32, 32, generator=generator) * 0.05
    model.fit([{"image": images, "label": torch.zeros(12)}], show_progress=False)
    return model


@pytest.fixture
def drift_reference():
    return np.random.default_rng(42).normal(size=(256, 16))
