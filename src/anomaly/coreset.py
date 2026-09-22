"""Greedy k-centre coreset subsampling.

The raw memory bank for one MVTec category is ~160k patches x 1536 dims, around 1GB in
float32. That is too large to ship, too slow to search, and mostly redundant: most patches of
a normal image look like most other patches of a normal image.

Random subsampling would preserve the *dense* regions of the feature distribution and drop the
sparse edges, which is exactly backwards -- the edges are where the decision boundary is. The
greedy k-centre (minimax facility location) objective instead picks points that minimise the
largest distance from any bank point to its nearest selected point, so it deliberately keeps
the outlying corners of the normal distribution.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from tqdm import tqdm


def _random_projection(features: Tensor, target_dim: int, generator: torch.Generator) -> Tensor:
    """Johnson-Lindenstrauss projection to speed up the greedy selection.

    The greedy loop costs O(N*k) distance computations; doing them in 1536 dimensions is the
    bottleneck. A random Gaussian projection preserves pairwise distances to within a small
    factor with high probability, which is accurate enough to *choose* points. The features
    that get stored are always the original full-dimensional ones, never the projected ones.
    """
    dim = features.shape[1]
    if target_dim >= dim:
        return features
    projection = torch.randn(dim, target_dim, generator=generator, dtype=features.dtype)
    projection /= target_dim**0.5
    return features @ projection


def greedy_coreset_indices(
    features: Tensor,
    ratio: float,
    seed: int,
    projection_dim: int = 128,
    show_progress: bool = True,
) -> Tensor:
    """Select ceil(ratio * N) representative rows of `features`; returns their indices.

    Args:
        features: (N, D) patch features.
        ratio: fraction of rows to keep, in (0, 1].
        seed: controls the projection and the starting point, so runs are reproducible.
        projection_dim: dimensionality used for distance computations during selection.
        show_progress: print a progress bar (off for tests).
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    if features.ndim != 2 or features.shape[0] == 0 or not torch.isfinite(features).all():
        raise ValueError("features must be a nonempty finite matrix")
    if projection_dim < 1:
        raise ValueError("projection_dim must be positive")
    n_samples = features.shape[0]
    n_select = max(1, math.ceil(ratio * n_samples))
    if n_select >= n_samples:
        return torch.arange(n_samples)

    generator = torch.Generator().manual_seed(seed)
    projected = _random_projection(features.float(), projection_dim, generator)

    # Start from the point furthest from the centroid: a deterministic, reproducible choice
    # that begins at an extreme of the distribution rather than wherever a random draw lands.
    centroid = projected.mean(dim=0, keepdim=True)
    start = int(torch.cdist(projected, centroid).squeeze(1).argmax())

    selected = torch.empty(n_select, dtype=torch.long)
    selected[0] = start
    # min_distances[i] = distance from point i to the nearest already-selected point.
    min_distances = torch.cdist(projected, projected[start : start + 1]).squeeze(1)
    # Identical descriptors must not make argmax choose the same row repeatedly.
    min_distances[start] = -torch.inf

    iterator = range(1, n_select)
    if show_progress:
        iterator = tqdm(iterator, desc="coreset", unit="pt", leave=False)

    for step in iterator:
        # The point currently worst-covered by the coreset is the one worth adding next.
        nxt = int(min_distances.argmax())
        selected[step] = nxt
        # Each new centre can only ever reduce a point's distance to its nearest centre, so
        # the running minimum updates in O(N) instead of recomputing against all centres.
        new_distances = torch.cdist(projected, projected[nxt : nxt + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, new_distances)
        min_distances[nxt] = -torch.inf

    return selected
