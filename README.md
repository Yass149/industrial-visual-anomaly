# Industrial visual anomaly detection

Detect and localise unusual regions in product images using a memory bank learned from normal images.

On a fixed MVTec AD holdout (seed 42; approximately 30% of each test defect type reserved for calibration):

| Category | Image AUROC | Pixel AUROC | AP | Missed defects | False alarms |
|---|---:|---:|---:|---:|---:|
| Bottle | 1.000 | 0.984 | 1.000 | 7/44 | 0/14 |
| Screw | 0.952 | 0.979 | 0.984 | 15/83 | 1/29 |
| Carpet | 0.998 | 0.988 | 0.999 | 1/62 | 2/20 |

These are cropped-frame holdout results, not the paper's full-test benchmark. Bottle illustrates why perfect ranking does not guarantee good threshold decisions. Confusion matrices, both AP and trapezoidal PR-AUC, prevalence-adjusted PR, split manifests and error grids are saved in `results/`.

Warmed CPU API medians were 118/117/124 ms for bottle/screw/carpet on Apple M4; the maximum was 301 ms across 60 sequential local TestClient requests. This includes decoding and overlays, but excludes startup and real network latency. It is not a load test. [Measured evidence](docs/assets/measured_run.json).

Reproduce with Python 3.11:

```bash
make reproduce
```

This installs pinned dependencies, checks/downloads data, trains, evaluates, benchmarks and runs lint/tests. First use downloads the dataset and backbone weights. Data defaults to `~/mvtec-data`. Timing and small floating-point differences depend on hardware. `make serve` opens the demo at `http://127.0.0.1:8000`; set its category in `configs/default.yaml`.

The [PatchCore-style method](https://arxiv.org/abs/2106.08265) extracts frozen WideResNet50-2 layer2/layer3 descriptors and retains a 1% greedy coreset. Nearest-normal distances produce scores and maps. A classifier would need defect labels; an autoencoder adds reconstruction training; PaDiM is a valid alternative with per-position distribution assumptions. This implementation is not validated for numerical parity with the authors' code. The coreset's accuracy cost versus a full bank remains unmeasured.

![Bottle holdout heatmap](docs/assets/bottle-overlay.png)

The threshold minimises `r × p × FNR + (1 − p) × FPR` on calibration images only. Defaults assume missed defects cost 10 times false alarms and defects occur in 2% of products. These are business assumptions. Deployment PR estimates reweight prevalence and assume stable class-conditional distributions. Small calibration sets make thresholds uncertain.

![Screw calibration cost sensitivity](docs/assets/cost-sensitivity.png)

Next: more independent calibration data, uncertainty estimates, full-frame inspection, and controlled coreset/backbone ablations. Centre cropping can hide border defects; the drift check is only a warning. Docker and CI definitions are included, but container execution and remote CI remain unverified. This is not production ready.

I would not again describe a method's benefits as established before evaluating failures, or show independently rescaled heatmaps as comparable evidence.

Start with the [visual guide](docs/START_HERE.html), then the [code map](docs/CODE_MAP.md) and [design choices](docs/DECISIONS.md). Dataset/derived-image terms: [MVTec CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad); [figure attribution](docs/assets/ATTRIBUTION.md). Raw dataset files are not committed.
