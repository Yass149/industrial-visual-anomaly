#!/usr/bin/env python3
"""Score a single image and write a heatmap overlay. The Phase 2 checkpoint.

Usage:
    python scripts/score_image.py --category bottle --image path/to/image.png
    python scripts/score_image.py --category bottle --sample broken_large
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly.config import load_config, resolve_device  # noqa: E402
from anomaly.data import build_display_transform, build_transform, load_image  # noqa: E402
from anomaly.patchcore import PatchCore  # noqa: E402
from anomaly.reproducibility import enable_determinism, set_seed  # noqa: E402
from anomaly.viz import overlay_heatmap  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument("--image", help="path to an image to score")
    parser.add_argument(
        "--sample", help="instead of --image, take the first test image of this defect type"
    )
    parser.add_argument("--out", default="results/overlay.png")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    enable_determinism()
    device = resolve_device(cfg.device)

    if args.image:
        image_path = Path(args.image)
    elif args.sample:
        candidates = sorted(
            (cfg.data.extract_dir / args.category / "test" / args.sample).glob("*.png")
        )
        if not candidates:
            print(f"no test images for defect type {args.sample!r}", file=sys.stderr)
            return 1
        image_path = candidates[0]
    else:
        print("pass --image or --sample", file=sys.stderr)
        return 1

    artifact = cfg.serve.artifacts_dir / f"{args.category}.pt"
    if not artifact.exists():
        print(f"no memory bank at {artifact}; run scripts/train.py first", file=sys.stderr)
        return 1
    model = PatchCore.load(artifact, cfg, device=device)

    original = load_image(image_path)
    tensor = build_transform(cfg.model.image_size, cfg.model.crop_size)(original).unsqueeze(0)

    started = time.perf_counter()
    result = model.score(tensor)
    latency_ms = (time.perf_counter() - started) * 1000

    score = float(result.scores[0])
    display: Image.Image = build_display_transform(cfg.model.image_size, cfg.model.crop_size)(
        original
    )
    overlay = overlay_heatmap(display, result.maps[0])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = cfg.eval.results_dir.parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)

    print(f"image     {image_path}")
    print(f"score     {score:.4f}")
    print(f"map range {float(result.maps[0].min()):.4f} - {float(result.maps[0].max()):.4f}")
    print(f"latency   {latency_ms:.0f} ms (bank {model.bank_size_mb:.1f} MB, device {device})")
    print(f"overlay   {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
