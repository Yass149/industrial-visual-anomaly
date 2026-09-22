"""Start the single-worker service using the same YAML as training and evaluation."""

from __future__ import annotations

import argparse

import uvicorn

from anomaly.api import create_app
from anomaly.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    uvicorn.run(create_app(cfg), host=cfg.serve.host, port=cfg.serve.port)


if __name__ == "__main__":
    main()
