"""Measure warmed batch-one CPU requests, including decode, overlay and local HTTP transport."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from fastapi.testclient import TestClient

from anomaly.api import create_app
from anomaly.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()
    if args.requests < 2:
        parser.error("--requests must be at least 2")
    cfg = load_config(args.config).model_copy(update={"device": "cpu"})
    results = []
    for category in cfg.data.categories:
        category_cfg = cfg.model_copy(
            update={"serve": cfg.serve.model_copy(update={"category": category})}
        )
        paths = sorted((cfg.data.extract_dir / category / "test").glob("*/*.png"))[: args.requests]
        requests_ms, processing_ms = [], []
        with TestClient(create_app(category_cfg)) as client:
            for i in range(args.requests):
                payload = paths[i % len(paths)].read_bytes()
                started = time.perf_counter()
                response = client.post(
                    "/predict", files={"file": ("sample.png", payload, "image/png")}
                )
                response.raise_for_status()
                requests_ms.append((time.perf_counter() - started) * 1000)
                processing_ms.append(response.json()["latency_ms"])
        results.append(
            {
                "category": category,
                "requests": args.requests,
                "request_median_ms": float(np.median(requests_ms)),
                "request_p95_ms": float(np.percentile(requests_ms, 95)),
                "request_max_ms": float(max(requests_ms)),
                "processing_median_ms": float(np.median(processing_ms)),
            }
        )
    report = {
        "device": "cpu",
        "scope": "warmed single sequential request via in-process TestClient; includes decode, transform, scoring, overlay, JSON; excludes real network and startup",
        "categories": results,
    }
    cfg.eval.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.eval.results_dir / "api_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
