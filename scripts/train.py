#!/usr/bin/env python3
"""Build a PatchCore memory bank for each configured category.

Usage:
    python scripts/train.py [--config configs/default.yaml] [--category bottle]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly.config import Config, load_config, resolve_device  # noqa: E402
from anomaly.data import MVTecDataset, collate_samples  # noqa: E402
from anomaly.patchcore import PatchCore  # noqa: E402
from anomaly.reproducibility import enable_determinism, set_seed  # noqa: E402


def train_category(cfg: Config, category: str, device: str, batch_size: int) -> dict[str, object]:
    """Fit a memory bank for one category and write it to the artifacts directory."""
    dataset = MVTecDataset(
        root=cfg.data.extract_dir,
        category=category,
        split="train",
        image_size=cfg.model.image_size,
        crop_size=cfg.model.crop_size,
    )
    # num_workers=0 keeps ordering and RNG state trivially reproducible; feature extraction
    # is the bottleneck here, not image loading.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    model = PatchCore(cfg, device=device)
    started = time.perf_counter()
    model.fit(loader)
    elapsed = time.perf_counter() - started

    destination = cfg.serve.artifacts_dir / f"{category}.pt"
    model.save(destination)

    summary = {
        "category": category,
        "train_images": len(dataset),
        "patches_total": model.metadata["n_patches_total"],
        "patches_kept": model.metadata["n_patches_kept"],
        "feature_dim": model.metadata["feature_dim"],
        "bank_mb": round(model.bank_size_mb, 2),
        "fit_seconds": round(elapsed, 1),
        "artifact": str(destination),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", help="train one category instead of all configured ones")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    enable_determinism()
    device = resolve_device(cfg.device)

    categories = [args.category] if args.category else list(cfg.data.categories)
    unknown = set(categories) - set(cfg.data.categories)
    if unknown:
        print(f"not in data.categories: {sorted(unknown)}", file=sys.stderr)
        return 1

    print(f"device={device} backbone={cfg.model.backbone} layers={cfg.model.layers}")
    summaries = [train_category(cfg, c, device, args.batch_size) for c in categories]

    cfg.serve.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.serve.artifacts_dir / "train_summary.json").write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
