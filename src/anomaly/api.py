"""One-category service with a calibrated, artifact-bound threshold."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from anomaly.config import Config, load_config, resolve_device
from anomaly.data import build_display_transform, build_transform
from anomaly.monitoring import DriftMonitor
from anomaly.patchcore import PatchCore
from anomaly.provenance import artifact_digest, scoring_signature
from anomaly.reproducibility import enable_determinism, set_seed
from anomaly.viz import overlay_heatmap, to_base64_png


class Prediction(BaseModel):
    category: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    heatmap_base64: str
    latency_ms: float
    monitoring: dict[str, object]


def create_app(cfg: Config | None = None) -> FastAPI:
    """Load artifacts once at startup; imports and unit tests do not download a backbone."""
    cfg = cfg or load_config(os.environ.get("ANOMALY_CONFIG", "configs/default.yaml"))
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_seed(cfg.seed)
        enable_determinism()
        category = cfg.serve.category
        artifact = cfg.serve.artifacts_dir / f"{category}.pt"
        calibration = json.loads(
            (cfg.serve.artifacts_dir / f"{category}.threshold.json").read_text()
        )
        if calibration["category"] != category or calibration["artifact_sha256"] != artifact_digest(
            artifact
        ):
            raise RuntimeError("threshold belongs to another artifact; run make evaluate")
        if calibration["scoring_signature"] != scoring_signature(cfg):
            raise RuntimeError("score settings changed since calibration; run make evaluate")
        if (calibration["cost_ratio"], calibration["base_rate"]) != (
            cfg.threshold.cost_ratio,
            cfg.threshold.base_rate,
        ):
            raise RuntimeError("business assumptions changed since calibration; run make evaluate")
        model = PatchCore.load(artifact, cfg, resolve_device(cfg.device))
        if model.reference_features is None:
            raise RuntimeError("artifact lacks monitoring reference; run make train evaluate")
        monitor = DriftMonitor(
            model.reference_features.numpy(),
            cfg.monitoring.window_size,
            cfg.monitoring.drift_alpha,
            cfg.monitoring.projections,
            cfg.seed,
        )
        # Warm up before becoming ready so the first user does not build the bank-neighbour cache.
        model.score(torch.zeros(1, 3, cfg.model.crop_size, cfg.model.crop_size))
        app.state.model, app.state.calibration, app.state.monitor = model, calibration, monitor
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="Industrial visual anomaly", lifespan=lifespan)
    transform = build_transform(cfg.model.image_size, cfg.model.crop_size)
    display_transform = build_display_transform(cfg.model.image_size, cfg.model.crop_size)

    @app.get("/health")
    def health() -> dict[str, object]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(503, "model not ready")
        return {"status": "ok", "category": cfg.serve.category}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return Path(__file__).with_name("frontend.html").read_text()

    def predict_bytes(payload: bytes, started: float) -> Prediction:
        try:
            with Image.open(io.BytesIO(payload)) as uploaded:
                # A compressed file can be small yet expand into a huge allocation.
                if uploaded.width * uploaded.height > 20_000_000:
                    raise ValueError("image exceeds 20 million pixels")
                uploaded.load()
                image = uploaded.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(
                422, "upload must be a decodable image of at most 20 million pixels"
            ) from exc
        tensor = transform(image).unsqueeze(0)
        # Forward hooks and the rolling monitor are mutable; serialize access in one worker.
        with lock:
            result = app.state.model.score(tensor)
            status = app.state.monitor.update(result.feature_summaries[0].numpy())
        calibration = app.state.calibration
        overlay = overlay_heatmap(
            display_transform(image),
            result.maps[0],
            vmin=calibration["heatmap_vmin"],
            vmax=calibration["heatmap_vmax"],
        )
        encoded = to_base64_png(overlay)
        score = float(result.scores[0])
        return Prediction(
            category=cfg.serve.category,
            anomaly_score=score,
            threshold=calibration["threshold"],
            is_anomaly=score >= calibration["threshold"],
            heatmap_base64=encoded,
            latency_ms=(time.perf_counter() - started) * 1000,
            monitoring=status,
        )

    @app.post("/predict", response_model=Prediction)
    async def predict(file: Annotated[UploadFile, File()]) -> Prediction:
        started = time.perf_counter()
        payload = await file.read(cfg.serve.max_upload_mb * 1024 * 1024 + 1)
        await file.close()
        if len(payload) > cfg.serve.max_upload_mb * 1024 * 1024:
            raise HTTPException(422, f"upload exceeds {cfg.serve.max_upload_mb} MB")
        return await run_in_threadpool(predict_bytes, payload, started)

    return app
