"""Save compact measured evidence and refresh the visual guide from local result files."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil

from anomaly.config import load_config, repo_root

DESCRIPTIONS = {
    "config.py": "Validate the single YAML file, resolve paths and choose a device.",
    "data.py": "Load normal/test samples, align masks and share deterministic transforms.",
    "features.py": "Extract frozen multi-layer image descriptors and flatten spatial patches.",
    "coreset.py": "Choose representative normal descriptors with a seeded greedy coverage algorithm.",
    "patchcore.py": "Fit, persist, load and score category banks; return maps and monitoring summaries.",
    "viz.py": "Normalise maps, blend aligned heatmaps and encode PNG responses.",
    "threshold.py": "Minimise prevalence-weighted false-alarm and missed-defect costs.",
    "evaluation.py": "Create disjoint split indices and calculate honest image-level metrics.",
    "monitoring.py": "Compare disjoint incoming windows with the training feature reference.",
    "provenance.py": "Fingerprint artifacts and scoring settings so thresholds cannot go stale silently.",
    "api.py": "Load the verified model once and serve health, frontend and image predictions.",
    "reproducibility.py": "Set Python/NumPy/torch seeds and deterministic-kernel preferences.",
    "download_data.py": "Fetch the archive, check its checksum, extract it and verify category structure.",
    "train.py": "Orchestrate normal-only fitting per category and save a training summary.",
    "score_image.py": "Score one file or example and save a relatively coloured heatmap.",
    "evaluate.py": "Calibrate thresholds and produce holdout metrics, manifests, maps and plots.",
    "benchmark_api.py": "Measure warmed sequential CPU requests through the actual API.",
    "serve.py": "Start a single service from YAML host, port and category settings.",
    "snapshot_report.py": "Refresh committed evidence, guide data and this file/function index.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg, root = load_config(args.config), repo_root()
    assets = root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    report = json.loads((cfg.eval.results_dir / "metrics.json").read_text())
    benchmark = json.loads((cfg.eval.results_dir / "api_benchmark.json").read_text())
    categories = []
    for metrics in report["categories"]:
        name = metrics["category"]
        records = json.loads((cfg.eval.results_dir / name / "predictions.json").read_text())
        categories.append(
            {
                "name": name,
                "metrics": metrics,
                "calibration": [
                    {"label": row["label"], "score": row["score"]}
                    for row in records
                    if row["split"] == "calibration"
                ],
            }
        )
    evidence = {
        "protocol": report["protocol"],
        "seed": report["seed"],
        "environment": report["environment"],
        "categories": categories,
        "api_benchmark": benchmark,
    }
    (assets / "measured_run.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    guide = root / "docs/START_HERE.html"
    html = re.sub(
        r'(<script id="evidence" type="application/json">).*?(</script>)',
        lambda match: match[1]
        + json.dumps(evidence, allow_nan=False).replace("<", "\\u003c")
        + match[2],
        guide.read_text(),
        flags=re.DOTALL,
    )
    guide.write_text(html)
    for source, destination in [
        ("bottle/example_overlay.png", "bottle-overlay.png"),
        ("screw/cost_sensitivity.png", "cost-sensitivity.png"),
        ("screw/false_negatives.png", "screw-false-negatives.png"),
    ]:
        shutil.copyfile(cfg.eval.results_dir / source, assets / destination)
    lines = [
        "# File and function map",
        "",
        "Use [the visual guide](START_HERE.html) for the big picture and [DESIGN.md](DESIGN.md) for mechanics.",
        "",
        "This index is generated from the current source by `make report`. Private helpers appear too.",
        "",
    ]
    for directory in ["src/anomaly", "scripts", "tests"]:
        for path in sorted((root / directory).glob("*.py")):
            relative = path.relative_to(root)
            tree = ast.parse(path.read_text())
            lines.extend(
                [
                    f"## [{relative}](../{relative})",
                    "",
                    DESCRIPTIONS.get(
                        path.name,
                        ast.get_docstring(tree) or "Package marker or automated regression checks.",
                    ),
                    "",
                ]
            )
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
                    description = (
                        doc.split("\n")[0]
                        if doc
                        else node.name.replace("_", " ").strip().capitalize()
                    )
                    lines.append(f"- `{node.name}` (line {node.lineno}): {description}")
            lines.append("")
    lines.extend(
        [
            "## Non-Python files",
            "",
            "- `configs/default.yaml`: category, preprocessing, model, cost, service and monitoring settings.",
            "- `src/anomaly/frontend.html`: plain drag-and-drop upload and prediction display.",
            "- `pyproject.toml`: package metadata, compatible dependency ranges, lint and test configuration.",
            "- `requirements.lock`: exact environment pins; update deliberately, then revalidate.",
            "- `Makefile`: the task entry points. `reproduce` runs stages sequentially.",
            "- `Dockerfile` / `.dockerignore`: container build, non-root worker and readiness check; build unverified locally.",
            "- `.github/workflows/ci.yml`: lint and offline tests on pushes and pull requests; remote execution unverified.",
            "- `.gitignore`: excludes datasets, banks, generated runtime results and local assistant state.",
            "- `README.md`: short measured summary, failures, reproduction and next steps.",
            "- `docs/START_HERE.html`: offline visual chapters with a calibration-only cost explorer and saved reading position.",
            "- `docs/DESIGN.md`: detailed technical walkthrough and limitations.",
            "- `docs/DECISIONS.md`: inherited/new choices and alternatives; no invented ablation claims.",
            "- `docs/RESUME.md`: quick return note, verification and operational commands.",
            "- `docs/assets/`: compact reference evidence and attributed diagnostic figures; not training data.",
            "- `artifacts/` (ignored): fitted banks, monitoring references and calibrated thresholds.",
            "- `results/` (ignored): metrics, exact split manifests, pixel predictions, diagnostics and benchmarks.",
            "- `.venv/` (ignored): installed local Python environment.",
            "",
        ]
    )
    (root / "docs/CODE_MAP.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
