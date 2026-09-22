import numpy as np
import pytest
import torch
from PIL import Image

from anomaly.coreset import greedy_coreset_indices
from anomaly.data import build_transform
from anomaly.patchcore import PatchCore


def test_preprocessing_is_identical_for_same_pixels():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 255, (80, 90, 3), dtype=np.uint8))
    transform = build_transform(40, 32)
    assert torch.equal(transform(image), transform(image.copy()))


def test_local_defect_scores_higher_and_localises(tiny_model):
    normal = torch.full((1, 3, 32, 32), 0.025)
    anomalous = normal.clone()
    anomalous[:, :, 8:16, 16:24] = 2.0
    good = tiny_model.score(normal)
    bad = tiny_model.score(anomalous)
    assert bad.scores.item() > good.scores.item() + 1
    peak = np.unravel_index(bad.maps[0].argmax().item(), (32, 32))
    assert 8 <= peak[0] < 16 and 16 <= peak[1] < 24
    assert bad.feature_summaries.shape == (1, 3)


def test_coreset_has_unique_indices_even_for_identical_points():
    features = torch.ones(10, 4)
    indices = greedy_coreset_indices(features, 0.35, 1, show_progress=False)
    assert len(indices) == 4  # ceil, not round
    assert len(indices.unique()) == 4


def test_coreset_covers_rare_normal_tail():
    features = torch.cat([torch.zeros(100, 2), torch.tensor([[10.0, 0.0], [20.0, 0.0]])])
    selected = greedy_coreset_indices(features, 0.02, 1, show_progress=False)
    assert torch.cdist(features, features[selected]).min(dim=1).values.max() == 0


def test_coreset_reproducible_projection():
    features = torch.randn(30, 12, generator=torch.Generator().manual_seed(5))
    a = greedy_coreset_indices(features, 0.2, 8, projection_dim=4, show_progress=False)
    b = greedy_coreset_indices(features, 0.2, 8, projection_dim=4, show_progress=False)
    assert torch.equal(a, b)


def test_empty_fit_and_anomalous_training_rejected(tiny_model):
    with pytest.raises(ValueError, match="empty"):
        tiny_model.fit([], show_progress=False)
    with pytest.raises(RuntimeError, match="normals"):
        tiny_model.fit(
            [{"image": torch.ones(1, 3, 32, 32), "label": torch.ones(1)}], show_progress=False
        )


def test_persistence_and_config_guard(tiny_model, cfg, tmp_path):
    path = tmp_path / "bank.pt"
    tiny_model.save(path)
    restored = PatchCore.load(path, cfg)
    image = torch.ones(1, 3, 32, 32)
    assert torch.equal(tiny_model.score(image).scores, restored.score(image).scores)
    assert torch.equal(restored.reference_features, tiny_model.reference_features)
    incompatible = cfg.model_copy(update={"model": cfg.model.model_copy(update={"crop_size": 24})})
    with pytest.raises(RuntimeError, match="crop_size"):
        PatchCore.load(path, incompatible)


def test_bank_neighbour_zero_is_self(tiny_model):
    tiny_model.bank = torch.ones(5, 3)
    tiny_model._bank_knn = None
    assert torch.equal(tiny_model._bank_neighbours()[:, 0], torch.arange(5))
