"""Seeding and determinism controls.

PatchCore has no stochastic training, but three things still draw on RNG: the coreset's
random projection, reference sampling, and the calibration split. Seeding all of them
is what lets the README claim a single command reproduces the numbers.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    """Ask torch for deterministic kernels where it has them.

    warn_only=True is deliberate: some convolution backward kernels have no deterministic
    implementation, and since nothing here is trained, refusing to run would cost more than
    the non-determinism it prevents. Forward-pass convolutions are deterministic already.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
