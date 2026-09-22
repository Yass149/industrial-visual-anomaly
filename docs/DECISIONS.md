# Why these choices, and when to choose something else

“Inherited” means the choice existed before this continuation. Reasons below explain technical
tradeoffs; they do not claim knowledge of an earlier author's private reasoning. “Added” means
implemented during this continuation. Alternatives are candidates, not measured comparisons.

| Decision | Reason for this project | Alternative and its cost | Status |
|---|---|---|---|
| PatchCore-style embeddings | Normal-only fitting and local distances are explainable | PaDiM stores per-position Gaussian statistics; compact but adds covariance estimation and positional assumptions | Inherited |
| No autoencoder | Avoid training a reconstruction model and relying on reconstruction error | Autoencoder can learn domain structure but needs capacity/training decisions and can reconstruct defects too well | Brief constraint |
| No supervised defect classifier | Training set contains no defect examples | Classifier is useful when representative labels exist; new defect types remain a concern | Brief constraint |
| Frozen ImageNet backbone | Reuse existing visual features, little training engineering | Domain fine-tuning can help but adds training and validation risk | Inherited |
| WideResNet50-2 | Strong conventional PatchCore baseline | ResNet18 reduces memory/latency, may lose representational power; no ablation yet | Inherited |
| Layer2 + layer3 | Local detail plus context | Earlier/later layers trade spatial resolution against context; optimum unmeasured | Inherited |
| Local average pooling | Smooth noisy local activations | Unpooled or learned aggregation retains detail but changes descriptor behaviour | Inherited |
| Concatenate aligned features | Simple, transparent descriptor construction | Original implementations may use additional feature pooling/reduction; numerical parity unverified | Inherited |
| Resize + crop | Bounded computation and shared train/serve geometry | Full-frame resize or tiling preserves borders, may distort or increase compute | Inherited, limitation retained |
| No augmentation | Reproducible baseline with fewer decisions | Realistic augmentation could improve invariance; unsuitable transformations can hide defects | Inherited |
| 1% greedy coreset | Small bank with coverage-oriented sampling | Random sampling cheaper; full bank more expensive; relative accuracy not measured | Inherited; uniqueness and ceil fixed |
| 128-D random projection | Faster representative selection | Full dimensional selection more expensive; projection error is a tradeoff | Inherited |
| Max + neighbour reweighting | A small defect can affect image score | Mean dilutes small defects; top-k reduces isolated noise, needs validation | Inherited; self-neighbour ordering fixed |
| Exact CPU distance lookup | Few thousand bank entries do not require another index library | FAISS/approximate retrieval can scale, but adds dependency and accuracy/reproducibility choices | Inherited |
| Bottle/screw/carpet | Rigid object, small changing pose, texture | Three object categories would match original wording more literally; would change experiment | Inherited |
| Fixed 30/70 calibration/evaluation | Prevent threshold tuning on evaluation labels | Normal-only calibration needs no labelled defects but cannot empirically optimise missed-defect cost | User selected |
| Stratify each defect type | Small defect groups appear on both sides | Group/time splits are preferable with batch/time metadata, absent here | Added |
| AP plus trapezoidal PR-AUC | Avoid silently conflating two PR summaries | AUROC alone hides precision problems at low prevalence | Added |
| Deployment prevalence weighting | Explicitly explore a rare-defect scenario | Benchmark precision alone uses an unrealistic prior; real production data is still needed | Added |
| Expected-cost threshold | Make missed defects vs false alarms explicit | F1 maximisation ignores business costs and prevalence assumptions | Brief requirement |
| Higher threshold on equal cost | Deterministic tie policy, fewer alarms | Lower threshold favours recall; choosing based on holdout would leak | Added; documented sensitivity |
| Fixed calibration colour bounds | Comparable colours across images of one category | Per-image min/max makes every normal image show a red maximum | Added |
| Artifact-bound thresholds | Refuse stale but plausible decisions | Loose config-only thresholds can be applied to incompatible scores | Added |
| One category per service | Clear API, one bank/reference/threshold | Multi-category routing adds product identification or explicit request routing | Inherited config, implemented |
| Locked inference | Protect mutable forward hooks and monitoring state | Thread-local models duplicate weights; redesigned stateless extraction can increase concurrency | Added |
| Image-level projected KS | Small, explainable distribution check | Patch-level tests exaggerate sample size; high-dimensional tests need more calibration and compute | Added |
| Disjoint 64-image windows | Avoid retesting almost identical recent windows | Rolling windows react sooner but increase dependence; time-wide false-alarm control still absent | Added |
| Offline unit + real integration tests | Fast CI with explicit real-data checks locally | Downloading 5GB in every CI run is costly and fragile | Added |
| Pin dependencies | Record the environment used for numbers | Floating versions ease upgrades but change behaviour silently | Inherited |
| Plain HTML frontend | Demonstrate inference without framework maintenance | React adds tooling without solving the core portfolio gap | Brief requirement |
| Docker + CI definitions | Make delivery steps concrete | Both still need their actual target environment exercised; configuration alone is not validation | Added, Docker/remote CI unverified |

See [DESIGN.md](DESIGN.md) for the mechanics and [the visual guide](START_HERE.html) for a shorter route.
