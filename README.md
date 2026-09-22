# Industrial visual anomaly detection

Detecting and localising manufacturing defects in product images using PatchCore, trained on
normal images only.

**Status: in progress.** This README is a placeholder. The real one is written last, from
measured results rather than intentions.

## Setup

```bash
make setup
```

This creates `.venv`, installs from `requirements.lock`, then downloads and verifies the
MVTec AD dataset (5.3 GB; expect a slow first run). The dataset path is `data.root` in
`configs/default.yaml` and defaults to `~/mvtec-data`, outside the repo.

MVTec AD is licensed for **non-commercial research use only**. No dataset files are committed.
