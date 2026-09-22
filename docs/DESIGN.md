# Design and data flow

A working document describing how this system operates, stage by stage, with real tensor
shapes and measured numbers. It is updated as each phase lands.

This is **not** the README. The README (Phase 8) is a short summary written last from final
results. This document is the long technical explanation: what every stage does, what shape
the data is at each point, and why each design choice was made over its alternatives.

**Status:** Phase 1 (skeleton) and Phase 2 (model) complete. Phases 3–8 pending.
Numbers marked _(pending)_ have not been measured yet and are not to be quoted.

---

## 1. The problem, and why it is framed as one-class

A factory has thousands of good units and a handful of bad ones. It cannot label defect types
in advance, because the next defect is by definition the one nobody anticipated. So supervised
classification is the wrong tool: there is no balanced labelled set to train on, and a
classifier trained on the defects you have seen will not generalise to the ones you have not.

The framing instead is **one-class**: learn what normal looks like from normal images only,
then flag anything that deviates. At training time the model sees **zero** anomalous images.
`MVTecDataset` enforces this — it raises if the train split contains an anomalous label.

Two things are required of the output:

1. An **image-level score** — is this unit defective?
2. A **pixel-level map** — where is the defect? Without this the system is unusable on a
   factory floor; an operator needs to see what the machine objected to.

---

## 2. End-to-end flow

### Training (once per category, minutes on CPU)

```
   ~209 normal PNGs (bottle/train/good/*.png)
            |
            |  data.build_transform  -- resize 256, centre-crop 224, ToTensor, ImageNet-normalise
            v
   (B, 3, 224, 224)  float32
            |
            |  features.PatchFeatureExtractor  -- FROZEN WideResNet50-2, forward hooks
            v
   layer2: (B,  512, 28, 28)        layer3: (B, 1024, 14, 14)
            |                                 |
            |  3x3 average pool, stride 1, padding 1  (both)
            v                                 v
   (B,  512, 28, 28)                 (B, 1024, 14, 14)
            |                                 |
            |                                 |  bilinear upsample to 28x28
            |                                 v
            +--------- concat on channels ----+
                            |
                            v
                  (B, 1536, 28, 28)           <- the patch feature map
                            |
                            |  features.flatten_patches
                            v
                  (B*784, 1536)               <- 784 = 28*28 patches per image
                            |
                            |  accumulate over the whole training set, on CPU
                            v
                  (~163856, 1536)  ~1.0 GB    <- the raw memory bank
                            |
                            |  coreset.greedy_coreset_indices, ratio 0.01
                            v
                  (~1639, 1536)    ~10 MB     <- the coreset memory bank, saved to disk
```

Nothing above is gradient descent. "Training" is: run normal images through a fixed function,
keep the outputs, throw away 99% of them intelligently.

### Inference (per image)

```
   one image
       |  same build_transform  (identical code path to training -- this matters, see 4.1)
       v
   (1, 3, 224, 224)
       |  same frozen extractor
       v
   (1, 1536, 28, 28)  ->  flatten  ->  (784, 1536)
       |
       |  torch.cdist against the bank (784, 1536) x (1639, 1536)
       v
   (784, 1639) distance matrix
       |
       |  min over bank dimension
       v
   (784,) per-patch distance to nearest normal patch,  (784,) index of that neighbour
       |                                        |
       |                                        +--> image score path (section 3.4)
       v
   reshape (1, 1, 28, 28)
       |  bilinear upsample to 224x224
       |  Gaussian blur, sigma 4 (kernel 33)
       v
   (224, 224) anomaly map  ---> heatmap overlay, and pixel-level AUROC in Phase 3
```

---

## 3. Stage by stage

### 3.1 Preprocessing — `data.py`

| step | operation | output |
|---|---|---|
| load | `Image.open(...).convert("RGB")` | PIL, e.g. 900×900 (bottle) or 1024×1024 |
| resize | bilinear to 256×256 | 256×256 |
| crop | centre crop 224 | 224×224 |
| tensor | `ToTensor()` | `(3, 224, 224)` in [0, 1] |
| normalise | ImageNet mean/std | `(3, 224, 224)` |

ImageNet statistics are used because the backbone is ImageNet-pretrained and frozen: its
features are only meaningful for inputs normalised the way it was trained.

**There is no augmentation, and that is a decision.** PatchCore models the distribution of
normal patch features. Augmenting the training set would widen that distribution with
variation the production camera never produces, making the memory bank tolerant of things it
should flag. Augmentation helps a discriminative classifier generalise; it actively hurts a
density model of normality.

**Known limitation:** resize-then-centre-crop discards the border. A defect at the extreme
edge of the frame can be cropped away entirely. This is the standard MVTec evaluation
protocol, kept for comparability with published numbers, but it is a real constraint that
would need revisiting for a deployment where parts are not centred.

Masks go through the same geometry with **nearest-neighbour** interpolation, then are
binarised (`> 0`), because bilinear interpolation of a 0/255 mask produces grey edge pixels
that are neither defect nor background.

Item order is sorted, not filesystem order, so index *i* is the same image on every machine.

### 3.2 Feature extraction — `features.py`

The backbone is `wide_resnet50_2` with ImageNet weights, `eval()`, `requires_grad_(False)`.
Features are captured with **forward hooks** on the named layers rather than by rebuilding a
truncated network — the hook approach means changing `model.layers` in the YAML requires no
code change.

**Why eval mode matters more than it appears.** It is not only about gradients. In train mode
BatchNorm normalises using *batch* statistics, so a patch's descriptor would depend on which
other images happened to share its batch — the same image would get different features
depending on batch composition, and the memory bank would be incoherent. Eval mode uses the
frozen ImageNet running statistics, making a descriptor a pure function of its own image.

**Why layer2 and layer3.**

| layer | channels | spatial (224 in) | why included / excluded |
|---|---|---|---|
| layer1 | 256 | 56×56 | excluded — edges and colour blobs, too generic to separate a defect from normal texture variation |
| layer2 | 512 | 28×28 | included — local structure, still spatially precise |
| layer3 | 1024 | 14×14 | included — larger-context structure |
| layer4 | 2048 | 7×7 | excluded — ImageNet-class-specific ("this is a dog"), and 7×7 is too coarse to localise anything useful |

**Neighbourhood pooling.** Each feature map is average-pooled with a 3×3 window, stride 1,
padding 1. This makes each descriptor summarise its local surroundings rather than a single
receptive field, which is what makes a nearest-neighbour lookup meaningful — an isolated
descriptor is too noisy to match reliably. The window is validated as odd in the config,
because an even window has no centre pixel and the pooled feature would not be aligned with
the patch position it claims to describe.

**Resolution alignment.** layer3's 14×14 map is bilinearly upsampled to layer2's 28×28 before
concatenation, so the two halves of a 1536-D descriptor describe the *same* patch. Concatenating
without aligning would produce descriptors that mix information from different image locations.

Result: **1536-D descriptor at each of 784 positions per image** (verified by measurement).

### 3.3 The memory bank and coreset — `coreset.py`

The raw bank for one category is roughly 163,856 × 1536 float32 ≈ **1.0 GB**. That is too
large to ship, too slow to search, and mostly redundant — most patches of a normal image look
like most other patches of a normal image.

**Why not random subsampling?** This is the central question about this stage. Random sampling
preserves the *dense* regions of the feature distribution and drops the sparse edges, which is
exactly backwards: the sparse edges are where the decision boundary sits. A rare-but-normal
patch that gets dropped becomes a guaranteed false positive at inference.

Greedy k-centre (minimax facility location) instead minimises the largest distance from any
bank point to its nearest kept point, so it deliberately keeps the outlying corners.

Measured on synthetic clustered data with a sparse tail (3 dense clusters of 2000 points plus
30 outliers, k = 60):

| method | max coverage distance | tail points kept |
|---|---|---|
| greedy k-centre | **1.393** | **6 / 30** |
| random sample (3 seeds) | 42.08 / 42.43 / 42.39 | 0 / 30 |

A caveat found while testing: on **iid Gaussian** data the coreset shows no advantage at all,
because in high dimensions all pairwise distances concentrate to the same value. The method
only pays off when the data has structure — which real patch features do, and synthetic noise
does not. Worth knowing before quoting the benefit as unconditional.

**The algorithm.**

1. Project to 128-D with a random Gaussian (Johnson–Lindenstrauss) matrix. This is a *speed*
   measure only: the greedy loop costs O(N·k) distance computations and 1536-D is the
   bottleneck. JL preserves pairwise distances well enough to *choose* points. The descriptors
   actually stored are always the original full-dimensional ones.
2. Start from the point furthest from the centroid — deterministic, and begins at an extreme
   of the distribution rather than wherever a random draw lands.
3. Maintain `min_distances[i]` = distance from point *i* to its nearest selected point.
   Repeatedly select `argmax(min_distances)` (the worst-covered point), then update with
   `minimum(min_distances, distance_to_new_point)`. Because a new centre can only ever *reduce*
   a point's distance to its nearest centre, the update is O(N) rather than a full recompute.

At ratio 0.01 the bank drops from ~1.0 GB to ~10 MB — the tradeoff that makes this servable.
Accuracy cost of the subsampling is measured in Phase 3 _(pending)_.

### 3.4 Scoring — `patchcore.py`

**Per-patch distance.** `torch.cdist` between the image's 784 descriptors and the K bank
entries, take the minimum over the bank axis. Chunked at 8192 rows so peak memory is bounded
regardless of batch or bank size.

**Pixel map.** The 784 distances reshape to 28×28, upsample bilinearly to 224×224, then get a
Gaussian blur (σ = 4, kernel 33). The blur does two things: it removes the blocky artefacts of
an 8× upsample, and it reflects the fact that a defect's influence is spatially continuous
rather than quantised to patch boundaries.

**Image score — max, not mean.** `s* = max_p min_m ‖p − m‖`. Taking the maximum is deliberate:
a small defect on an otherwise normal image would be averaged away by a mean. The cost is
sensitivity to a single noisy patch; if that shows up in Phase 3 (screw is the likely
candidate) the alternative is the mean of the top-k patch distances, which trades some
sensitivity for robustness.

**Neighbourhood reweighting.** `s*` is scaled by how isolated its matched bank entry `m*` is:

> If `m*`'s own nearest neighbours in the bank are far from the test patch, `m*` sits in a
> sparse region of the normal distribution. It is a rare normal example, so matching it is weak
> evidence of normality and the score stays high. If the whole neighbourhood is equally close,
> the match is well supported and the score is damped.

The paper writes this as `w = 1 − exp(d₀) / Σⱼ exp(dⱼ)`. **That expression overflows** at the
distance magnitudes these features produce. It is algebraically exactly `1 − softmax(d)[0]`, so
the implementation uses `torch.softmax` and inherits its shift-invariant stabilisation for free.

### 3.5 Persistence

A saved artifact is `{bank: Tensor, metadata: dict}` via `torch.save`, loaded with
`weights_only=True`. `PatchCore.load` **refuses** a bank whose stored `backbone`, `layers`,
`patch_neighbourhood`, `image_size` or `crop_size` disagree with the current config. Scoring
freshly-extracted features against a stale bank would produce plausible-looking nonsense,
which is the worst failure mode available — wrong numbers that look right.

---

## 4. Cross-cutting decisions

### 4.1 One preprocessing path

`build_transform` is called by training, by evaluation, and (Phase 5) by the API. If the
service preprocessed differently from evaluation, the served model would not be the evaluated
model and every number in the README would be false. This is the single most common way a
portfolio project quietly breaks, so preprocessing lives in exactly one function.

### 4.2 Config as the only variable surface

Every knob is in `configs/default.yaml`, validated by Pydantic with `extra="forbid"` and
`frozen=True`. A typo fails loudly at the offending key rather than silently leaving a default
in place. Validators encode real invariants, not just types: categories must be real MVTec
names, `crop_size ≤ image_size`, `patch_neighbourhood` must be odd, and `serve.category` must
be one a memory bank actually exists for. All twelve failure modes were exercised and each
rejects with the key named.

Adding a fourth category is one string in `data.categories`. No code change.

### 4.3 Reproducibility, and the limit of it

Seeds are set for Python, NumPy and torch; `use_deterministic_algorithms(warn_only=True)` is
on. Paths are absolutised against the repo root rather than the working directory.

**The honest limit:** scoring the same image alone versus in a batch of two gives scores
differing by relative **7e-8**. Float32 accumulation order changes with batch size inside the
BLAS and convolution kernels. This is not a bug and is not fixable at reasonable cost; it means
determinism tests must use `allclose` for scores (exact equality does hold for preprocessing,
and for save/load round-trips, both verified).

### 4.4 Device policy

`device: auto` resolves to CUDA if present, else CPU. Apple MPS must be requested explicitly:
several ops fall back to CPU silently and its reductions are not bit-reproducible, which would
undercut the reproducibility claim. CPU is the supported path, because a reviewer will clone
this on a laptop.

---

## 5. Measured numbers

Synthetic-data validation, Phase 2 (M4 CPU):

| quantity | value |
|---|---|
| patch feature map | `(B, 1536, 28, 28)` |
| descriptors per image | 784 |
| single-image scoring latency | **67 ms** |
| feature extraction | ~80 ms / image |
| defect localisation | map peak at (119, 80) for a blob centred at (120, 80) |
| score separation | 7.4× anomalous vs normal |
| save/load | bit-identical bank and scores |

Real MVTec numbers (per-category bank size, fit time, AUROC, PR-AUC, latency): _(pending —
Phase 3)_.

---

## 6. Repository layout

```
configs/default.yaml            every tunable parameter
src/anomaly/
  config.py                     typed, validated config loader
  data.py                       MVTecDataset + the single preprocessing path
  features.py                   frozen backbone -> patch descriptors
  coreset.py                    greedy k-centre subsampling
  patchcore.py                  memory bank, scoring, persistence
  viz.py                        heatmap overlay, base64 PNG
  reproducibility.py            seeding and determinism
scripts/
  download_data.py              fetch + checksum + extract MVTec AD
  train.py                      build a bank per category
  score_image.py                score one image, write an overlay
docs/DESIGN.md                  this document
```

### Dependency policy

`pyproject.toml` declares compatible *ranges* — the contract. `requirements.lock` pins 60
exact versions — what every reported number was produced with. `make install` installs from
the lock, then `pip install --no-deps -e .`; without `--no-deps`, pip re-resolves and can
quietly upgrade a pinned transitive dependency straight back out of the environment.

`src/` layout is deliberate: with a flat layout `import anomaly` resolves to the working
directory whether or not the package installed correctly, so tests pass locally and the
Docker build fails. Under `src/`, the only way to import it is to install it.

### Dataset provenance — stated plainly

The official MVTec AD archive sits behind a licence-acceptance form with no stable direct URL.
`download_data.py` pulls a HuggingFace mirror, verified to contain the original directory tree
(`<category>/{train,test,ground_truth}/`, plus MVTec's own licence and readme files), and
checks it against a SHA-256 digest pinned in the config.

**That digest is self-generated, not published by MVTec.** It guarantees everyone gets bytes
identical to those that produced the reported numbers. It does *not* attest that the mirror
equals the original. A reviewer with their own licensed copy can point `data.root` at it.

MVTec AD is licensed for **non-commercial research use only**. No dataset files are committed.

---

## 7. Phase status

| phase | state |
|---|---|
| 1. Skeleton | done — config, lock file, download script, `make setup` |
| 2. Model | code done, validated on synthetic data; real-data checkpoint pending download |
| 3. Evaluation | not started |
| 4. Threshold / cost model | not started |
| 5. Serving | not started |
| 6. Tests + CI | not started — `make test` currently fails with no tests collected |
| 7. Drift monitoring | not started |
| 8. README | not started, by design |

## 8. Open questions and known limitations

- Border defects can be cropped away by the 256→224 centre crop (section 3.1).
- Image score uses max-patch-distance; may need top-k mean if screw underperforms (3.4).
- Coreset accuracy cost has not yet been measured against the full bank (3.3).
- Batch-size-dependent float32 variation at relative 7e-8 (4.3).
- Dataset provenance is a mirror with a self-generated checksum (section 6).
