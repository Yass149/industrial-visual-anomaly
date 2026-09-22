# Return here next time

**Start:** open `docs/START_HERE.html` in a browser. It shows one chapter at a time and remembers
the last chapter locally. There is a real-calibration cost explorer in chapter 7. You can also
run `make guide` and visit http://127.0.0.1:8765/docs/START_HERE.html.

## Where the project stands

Phases 1–8 have implementations. The original PatchCore-style model was retained and corrected;
evaluation, threshold optimisation, FastAPI/frontend, regression tests, CI/Docker definitions,
monitoring and measured documentation were added. Local datasets and trained artifacts remain
outside git; regenerate them with the commands below after cloning.

The bank uses normal training images only. The **user-approved** threshold protocol reserves
about 30% of labelled MVTec test images within each defect type for calibration and reports
performance on the remaining 70%. Seed: 42. Do not tune further on that holdout and continue
calling it untouched evidence for new decisions.

The main result is not “everything works perfectly”: bottle misses 7/44 defects at its
calibrated threshold despite AUROC 1.000; screw misses 15/83; carpet misses 1/62. Review the
false-negative grids. Default assumptions: missed-defect/false-alarm cost 10, defect rate 2%.

## Local verification (2026-09-22)

`make setup`, ruff and black passed. **23 offline tests and 4 real-data integration tests passed**
with no skips in the integration run. The live HTTP service returned 200 for health, frontend
and a real bottle upload, and 422 for malformed image bytes. The visual guide was opened and
inspected in Safari. Docker and remote CI remain unverified as described below.

## Commands, from the project root

| Want to… | Command |
|---|---|
| Use the existing local demo | `make serve` → http://127.0.0.1:8000 |
| Run offline regression checks | `make test` |
| Check real images and API with current artifacts | `make integration` |
| Recompute held-out metrics and threshold files | `make evaluate` |
| Rebuild memory banks after a model/config change | `make train` then `make evaluate` |
| Measure local CPU service latency | `make benchmark` |
| Refresh compact evidence / guide / function map | `make report` |
| Reproduce all stages | `make reproduce` |
| Check formatting and lint | `make lint` |

`make reproduce` overwrites local trained artifacts/results and refreshes documentation
assets. It downloads data only if absent. Allow roughly 10 GB for the archive plus extraction,
plus weights, packages and transient training memory. The CPU training run here took minutes.
`make clean` removes artifacts/results; `make clean-data` removes the legacy repo-local data
path, not the default external dataset directory. Do not delete the external dataset casually.

## A concrete request

Start `make serve`, then in another terminal:

```bash
curl -F 'file=@/absolute/path/to/a/bottle.png' http://127.0.0.1:8000/predict
```

The JSON's `heatmap_base64` is a PNG. The frontend displays it automatically. Upload only the
configured category; the system does not identify product type. Change `serve.category` in
the YAML to `screw` or `carpet` to run one of their already calibrated banks.

## Container instructions (not executed here)

Docker is not installed on this machine. With Docker installed, after `make train evaluate`:

```bash
docker build -t industrial-anomaly .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/artifacts:/app/artifacts:ro" industrial-anomaly
```

Default backbone weights are fetched during the image build. The container listens on all
interfaces internally, while this port mapping exposes it only on the local host. Changing
the backbone requires making those weights available too. Run and validate this before claiming
container portability. GitHub Actions is configured; no remote CI run was performed here.

## Evidence and boundaries

- `artifacts/*.pt`: trained bank + image-level drift reference + training metadata.
- `artifacts/*.threshold.json`: calibration decision tied to the exact artifact and settings.
- `results/metrics.json`: all measured category metrics and environment/config.
- `results/<category>/predictions.json`: exact image split and scores.
- `results/<category>/pixel_predictions.npz`: holdout pixel maps and masks.
- `results/api_benchmark.json`: warmed local TestClient latency, 20 requests/category.
- `docs/assets/measured_run.json`: compact reference-run snapshot without private absolute paths.

The guide, plots and threshold explorer use real saved results. Illustrative numeric examples
in DESIGN.md are explicitly labelled. The original design document contained unverified
synthetic measurements and overbroad claims; those are not promoted as current evidence.

Known boundaries: centre-crop blind spots, small calibration normal sets, correlated-image
risk, no uncertainty intervals, unmeasured coreset/backbone ablations, no high-concurrency
load test, finite projected drift sensitivity, no production security layer. A few external
Starlette/httpx deprecation warnings occur with the pinned environment.

## One manageable next step

Open chapter 6 of the guide. Choose one missed screw image. Explain these four things:
what pixels survived preprocessing; where the anomaly map is high; what the image score is;
why the fixed threshold did not flag it. Use the code map only for the function you need.
