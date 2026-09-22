#!/usr/bin/env python3
"""Score once, calibrate on a fixed subset, and report untouched holdout performance."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from anomaly.config import Config, load_config, resolve_device
from anomaly.data import MVTecDataset, build_display_transform, load_image
from anomaly.evaluation import image_metrics, split_indices
from anomaly.patchcore import PatchCore
from anomaly.provenance import artifact_digest, scoring_signature
from anomaly.reproducibility import enable_determinism, set_seed
from anomaly.threshold import optimise_threshold
from anomaly.viz import overlay_heatmap


def write_json(path: Path, value: object) -> None:
    """Reject nonstandard JSON rather than shipping NaN thresholds to the service."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_errors(
    cfg: Config,
    dataset: MVTecDataset,
    labels: np.ndarray,
    scores: np.ndarray,
    maps: np.ndarray,
    holdout: np.ndarray,
    threshold: float,
    vmax: float,
    directory: Path,
) -> None:
    """Show actual errors only, with common colour bounds and ground-truth contours."""
    display = build_display_transform(cfg.model.image_size, cfg.model.crop_size)
    fp = [i for i in holdout if labels[i] == 0 and scores[i] >= threshold]
    fn = [i for i in holdout if labels[i] == 1 and scores[i] < threshold]
    groups = {
        "false_positives": sorted(fp, key=lambda i: -scores[i]),
        "false_negatives": sorted(fn, key=lambda i: scores[i]),
    }
    for name, indices in groups.items():
        selected = indices[: cfg.eval.n_qualitative]
        count = max(1, len(selected))
        columns = min(4, count)
        rows = (count + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.7 * rows), squeeze=False)
        if not selected:
            axes[0, 0].text(
                0.5, 0.5, "No errors of this type\non this holdout", ha="center", va="center"
            )
        for ax, i in zip(axes.flat, selected, strict=False):
            sample = dataset[int(i)]
            image = display(load_image(sample.path))
            ax.imshow(overlay_heatmap(image, maps[i], vmin=0, vmax=vmax))
            if sample.mask.any():
                ax.contour(sample.mask.numpy(), levels=[0.5], colors="white", linewidths=0.7)
            relative = Path(sample.path).relative_to(dataset.category_dir)
            ax.set_title(f"{relative}\nscore {scores[i]:.3f}", fontsize=9)
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(
            f"{dataset.category_dir.name}: {name.replace('_', ' ')} | threshold {threshold:.3f}"
        )
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=130)
        plt.close(fig)
    # A deterministic holdout defect is the README illustration, never selected by aesthetics.
    index = next(int(i) for i in holdout if labels[i] == 1)
    overlay_heatmap(display(load_image(dataset[index].path)), maps[index], vmin=0, vmax=vmax).save(
        directory / "example_overlay.png"
    )


def evaluate_category(cfg: Config, category: str) -> dict[str, object]:
    """Create metrics, split manifest, calibrated threshold, and diagnostic figures."""
    device = resolve_device(cfg.device)
    artifact = cfg.serve.artifacts_dir / f"{category}.pt"
    model = PatchCore.load(artifact, cfg, device)
    dataset = MVTecDataset(
        cfg.data.extract_dir, category, "test", cfg.model.image_size, cfg.model.crop_size
    )
    calibration, holdout = split_indices(
        [item[2] for item in dataset.items], cfg.eval.calibration_fraction, cfg.seed
    )
    directory = cfg.eval.results_dir / category
    directory.mkdir(parents=True, exist_ok=True)
    scores, labels, latencies, maps, masks = [], [], [], [], []
    # Warm-up includes the lazily cached bank neighbourhood search; report it separately.
    started = time.perf_counter()
    model.score(dataset[0].image.unsqueeze(0))
    warmup_ms = (time.perf_counter() - started) * 1000
    for sample in tqdm(dataset, desc=f"evaluating {category}"):
        started = time.perf_counter()
        result = model.score(sample.image.unsqueeze(0))
        latencies.append((time.perf_counter() - started) * 1000)
        scores.append(float(result.scores[0]))
        labels.append(sample.label)
        maps.append(result.maps[0].numpy())
        masks.append(sample.mask.numpy().astype(np.uint8))
    scores, labels = np.asarray(scores), np.asarray(labels)
    maps, masks = np.asarray(maps), np.asarray(masks)
    point = optimise_threshold(
        labels[calibration], scores[calibration], cfg.threshold.cost_ratio, cfg.threshold.base_rate
    )
    # Heatmap bounds use calibration maps only, so colours in holdout and API are comparable.
    vmax = max(float(maps[calibration].max()), 1e-12)
    threshold_record = {
        **point.to_dict(),
        "category": category,
        "artifact_sha256": artifact_digest(artifact),
        "scoring_signature": scoring_signature(cfg),
        "heatmap_vmin": 0.0,
        "heatmap_vmax": vmax,
        "calibration_images": len(calibration),
    }
    write_json(cfg.serve.artifacts_dir / f"{category}.threshold.json", threshold_record)
    metrics = image_metrics(
        labels[holdout],
        scores[holdout],
        point.threshold,
        cfg.threshold.cost_ratio,
        cfg.threshold.base_rate,
    )
    metrics.update(
        {
            "category": category,
            "calibration_images": len(calibration),
            "pixel_auroc": float(roc_auc_score(masks[holdout].ravel(), maps[holdout].ravel())),
            "calibration": point.to_dict(),
            "bank_mb": model.bank_size_mb,
            "score_latency_ms": {
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "max": float(max(latencies)),
                "warmup": warmup_ms,
            },
            "latency_scope": "batch=1 model.score only; excludes decode, transforms, overlay, HTTP",
            "device": device,
            "artifact_sha256": threshold_record["artifact_sha256"],
        }
    )
    manifest = [
        {
            "path": str(Path(path).relative_to(cfg.data.extract_dir)),
            "label": label,
            "defect_type": defect,
            "split": "calibration" if i in set(calibration) else "evaluation",
            "score": float(scores[i]),
        }
        for i, (path, label, defect) in enumerate(dataset.items)
    ]
    write_json(directory / "predictions.json", manifest)
    write_json(directory / "metrics.json", metrics)
    np.savez_compressed(
        directory / "pixel_predictions.npz",
        indices=holdout,
        scores=maps[holdout],
        masks=masks[holdout],
    )
    sweep = [
        optimise_threshold(
            labels[calibration], scores[calibration], ratio, cfg.threshold.base_rate
        ).to_dict()
        for ratio in cfg.threshold.cost_ratio_sweep
    ]
    write_json(directory / "cost_sensitivity.json", sweep)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ratios = [p["cost_ratio"] for p in sweep]
    axes[0].step(ratios, [p["threshold"] for p in sweep], where="post")
    axes[0].set(ylabel="Chosen score threshold", title="Calibration threshold")
    axes[1].plot(
        ratios, [p["false_negative_rate"] for p in sweep], "o-", label="Missed defects (FNR)"
    )
    axes[1].plot(
        ratios, [p["false_positive_rate"] for p in sweep], "s-", label="False alarms (FPR)"
    )
    axes[1].set(ylabel="Rate", ylim=(-0.03, 1.03), title="Calibration operating rates")
    axes[1].legend()
    for ax in axes:
        ax.set(xscale="log", xlabel="Missed-defect cost / false-alarm cost")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{category} | assumed defect rate {cfg.threshold.base_rate:.0%} | calibration only"
    )
    fig.tight_layout()
    fig.savefig(directory / "cost_sensitivity.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, name in [
        ("pr_curve", "Holdout prevalence"),
        (
            "scenario_pr_curve",
            "2% scenario" if cfg.threshold.base_rate == 0.02 else "Deployment scenario",
        ),
    ]:
        curve = metrics[key]
        ax.step(curve["recall"], curve["precision"], where="post", label=name)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=f"{category}: holdout PR",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "precision_recall.png", dpi=140)
    plt.close(fig)
    plot_errors(cfg, dataset, labels, scores, maps, holdout, point.threshold, vmax, directory)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    enable_determinism()
    metrics = [evaluate_category(cfg, category) for category in cfg.data.categories]
    report = {
        "protocol": (
            "30% calibration by defect type; 70% untouched evaluation"
            if cfg.eval.calibration_fraction == 0.3
            else "fixed calibration/evaluation split by defect type"
        ),
        "seed": cfg.seed,
        "calibration_fraction": cfg.eval.calibration_fraction,
        "base_rate": cfg.threshold.base_rate,
        "cost_ratio": cfg.threshold.cost_ratio,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
        },
        "config": cfg.model_dump(mode="json"),
        "categories": metrics,
    }
    write_json(cfg.eval.results_dir / "metrics.json", report)
    print(
        json.dumps(
            [
                {
                    k: m[k]
                    for k in (
                        "category",
                        "image_auroc",
                        "pixel_auroc",
                        "average_precision",
                        "confusion_matrix",
                    )
                }
                for m in metrics
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
