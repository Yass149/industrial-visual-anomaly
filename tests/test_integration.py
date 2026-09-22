"""Opt-in evidence against real local data; CI tests do not download a 5GB dataset."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly.api import create_app
from anomaly.config import load_config
from anomaly.data import build_transform, load_image
from anomaly.patchcore import PatchCore


@pytest.mark.integration
@pytest.mark.parametrize(
    "category,defect", [("bottle", "broken_large"), ("screw", "scratch_head"), ("carpet", "color")]
)
def test_real_fixture_ordering(category, defect):
    cfg = load_config()
    artifact = cfg.serve.artifacts_dir / f"{category}.pt"
    if not artifact.exists() or not cfg.data.extract_dir.exists():
        pytest.skip("run make train with the local MVTec dataset first")
    transform = build_transform(cfg.model.image_size, cfg.model.crop_size)
    model = PatchCore.load(artifact, cfg, "cpu")
    scores = []
    for kind in ["good", defect]:
        path = sorted((cfg.data.extract_dir / category / "test" / kind).glob("*.png"))[0]
        scores.append(model.score(transform(load_image(path)).unsqueeze(0)).scores.item())
    assert scores[1] > scores[0]


@pytest.mark.integration
def test_real_api_end_to_end():
    cfg = load_config()
    if not (cfg.serve.artifacts_dir / f"{cfg.serve.category}.threshold.json").exists():
        pytest.skip("run make evaluate first")
    sample = sorted((cfg.data.extract_dir / cfg.serve.category / "test" / "good").glob("*.png"))[0]
    # Encode a real image through the same upload path as the browser.
    buffer = io.BytesIO()
    Image.open(sample).save(buffer, format="PNG")
    with TestClient(create_app(cfg)) as client:
        result = client.post(
            "/predict", files={"file": ("real.png", buffer.getvalue(), "image/png")}
        )
    assert result.status_code == 200
    assert result.json()["latency_ms"] > 0
