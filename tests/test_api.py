import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly.api import create_app
from anomaly.provenance import artifact_digest, scoring_signature


@pytest.fixture
def app(cfg, tiny_model):
    artifact = cfg.serve.artifacts_dir / "bottle.pt"
    tiny_model.save(artifact)
    threshold = {
        "category": "bottle",
        "threshold": 1.0,
        "cost_ratio": cfg.threshold.cost_ratio,
        "base_rate": cfg.threshold.base_rate,
        "artifact_sha256": artifact_digest(artifact),
        "scoring_signature": scoring_signature(cfg),
        "heatmap_vmin": 0.0,
        "heatmap_vmax": 4.0,
    }
    (cfg.serve.artifacts_dir / "bottle.threshold.json").write_text(json.dumps(threshold))
    return create_app(cfg)


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (60, 50), (128, 128, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_api_schema_overlay_and_bad_uploads(app):
    with TestClient(app) as client:
        assert client.get("/health").json()["category"] == "bottle"
        assert "Inspect a product" in client.get("/").text
        response = client.post(
            "/predict", files={"file": ("sample.png", image_bytes(), "image/png")}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["is_anomaly"] == (result["anomaly_score"] >= result["threshold"])
        assert result["latency_ms"] > 0
        assert Image.open(io.BytesIO(base64.b64decode(result["heatmap_base64"]))).size == (32, 32)
        assert result["monitoring"]["pending_images"] == 1
        for payload in [b"not an image", b"", b"x" * (1024 * 1024 + 1)]:
            assert (
                client.post(
                    "/predict", files={"file": ("bad.png", payload, "image/png")}
                ).status_code
                == 422
            )
        assert client.post("/predict").status_code == 422


def test_stale_threshold_refused(app, cfg):
    path = cfg.serve.artifacts_dir / "bottle.threshold.json"
    record = json.loads(path.read_text())
    record["artifact_sha256"] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="another artifact"), TestClient(app):
        pass
