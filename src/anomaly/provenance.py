"""Bind calibrated thresholds to the exact artifact and score configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anomaly.config import Config


def artifact_digest(path: Path) -> str:
    """Hash in chunks so fingerprints do not duplicate an artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scoring_signature(cfg: Config) -> str:
    """Thresholds are invalid if preprocessing or scoring settings have changed."""
    settings = {"model": cfg.model.model_dump(), "implementation": "patchcore-v2"}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
