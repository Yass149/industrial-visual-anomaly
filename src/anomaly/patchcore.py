"""PatchCore: a memory bank of normal patch features, scored by nearest-neighbour distance.

Training is not gradient descent. It is: run every normal training image through a frozen
backbone, keep the patch descriptors, subsample them to a coreset. Scoring a test image is:
extract its patch descriptors, measure each one's distance to the nearest stored normal
descriptor, and take the map of those distances as the anomaly map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from anomaly.config import Config
from anomaly.coreset import greedy_coreset_indices
from anomaly.features import PatchFeatureExtractor, flatten_patches

# Cap on rows compared against the bank in one cdist call, to bound peak memory regardless
# of batch size or bank size.
_DISTANCE_CHUNK = 8192


@dataclass
class ScoreResult:
    """Per-image anomaly score and the pixel map it was derived from."""

    scores: Tensor  # (B,) raw distances, higher is more anomalous
    maps: Tensor  # (B, crop, crop) upsampled, smoothed per-pixel distances


class PatchCore:
    """Memory-bank anomaly detector. Fit on normal images, scored by distance to the bank."""

    def __init__(self, cfg: Config, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.extractor = PatchFeatureExtractor(
            backbone=cfg.model.backbone,
            layers=cfg.model.layers,
            neighbourhood=cfg.model.patch_neighbourhood,
            device=device,
        )
        self.bank: Tensor | None = None
        self.feature_shape: tuple[int, int] | None = None
        self.metadata: dict[str, Any] = {}
        self._bank_knn: Tensor | None = None

    # ---------------------------------------------------------------- fitting

    def fit(self, loader: DataLoader, show_progress: bool = True) -> None:
        """Build the memory bank from a loader over normal images only."""
        chunks: list[Tensor] = []
        n_images = 0
        batches = tqdm(loader, desc="extracting", unit="batch") if show_progress else loader

        for batch in batches:
            images = batch["image"]
            if int(batch["label"].sum()) != 0:
                raise RuntimeError("fit() received an anomalous image; train on normals only")
            features = self.extractor(images)
            if self.feature_shape is None:
                self.feature_shape = (int(features.shape[2]), int(features.shape[3]))
            # Accumulate on CPU: the full bank is ~1GB before subsampling and does not need
            # to sit on an accelerator, where it would likely not fit alongside the backbone.
            chunks.append(flatten_patches(features).cpu())
            n_images += images.shape[0]

        all_patches = torch.cat(chunks)
        del chunks

        indices = greedy_coreset_indices(
            all_patches,
            ratio=self.cfg.model.coreset_ratio,
            seed=self.cfg.seed,
            show_progress=show_progress,
        )
        self.bank = all_patches[indices].contiguous()
        self._bank_knn = None

        self.metadata = {
            "backbone": self.cfg.model.backbone,
            "layers": list(self.cfg.model.layers),
            "patch_neighbourhood": self.cfg.model.patch_neighbourhood,
            "image_size": self.cfg.model.image_size,
            "crop_size": self.cfg.model.crop_size,
            "coreset_ratio": self.cfg.model.coreset_ratio,
            "n_neighbours": self.cfg.model.n_neighbours,
            "gaussian_blur_sigma": self.cfg.model.gaussian_blur_sigma,
            "seed": self.cfg.seed,
            "n_train_images": n_images,
            "n_patches_total": int(all_patches.shape[0]),
            "n_patches_kept": int(self.bank.shape[0]),
            "feature_dim": int(self.bank.shape[1]),
            "feature_shape": list(self.feature_shape or ()),
        }

    # ---------------------------------------------------------------- scoring

    def _nearest_distances(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        """For each row of `patches`, distance to and index of its nearest bank entry."""
        assert self.bank is not None
        bank = self.bank.to(patches.device)
        values: list[Tensor] = []
        indices: list[Tensor] = []
        for start in range(0, patches.shape[0], _DISTANCE_CHUNK):
            block = patches[start : start + _DISTANCE_CHUNK]
            distance = torch.cdist(block, bank)
            best, best_index = distance.min(dim=1)
            values.append(best)
            indices.append(best_index)
        return torch.cat(values), torch.cat(indices)

    def _bank_neighbours(self) -> Tensor:
        """(K, b) indices of each bank entry's b nearest bank entries, itself first."""
        assert self.bank is not None
        if self._bank_knn is None:
            b = min(self.cfg.model.n_neighbours, self.bank.shape[0])
            distance = torch.cdist(self.bank, self.bank)
            self._bank_knn = distance.topk(b, dim=1, largest=False).indices
        return self._bank_knn

    @torch.no_grad()
    def score(self, images: Tensor) -> ScoreResult:
        """Score a batch of preprocessed images."""
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("model has no memory bank; call fit() or load() first")

        features = self.extractor(images)
        batch = features.shape[0]
        height, width = self.feature_shape
        patches = flatten_patches(features).cpu()

        distances, nn_index = self._nearest_distances(patches)
        patch_distances = distances.view(batch, height * width)

        image_scores = self._reweighted_image_scores(
            patches.view(batch, height * width, -1),
            patch_distances,
            nn_index.view(batch, height * width),
        )

        maps = patch_distances.view(batch, 1, height, width)
        maps = F.interpolate(
            maps,
            size=(self.cfg.model.crop_size, self.cfg.model.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        sigma = self.cfg.model.gaussian_blur_sigma
        if sigma > 0:
            # Upsampling a 28x28 map to 224x224 leaves blocky artefacts; smoothing also
            # reflects that a defect's influence is spatially continuous, not patch-quantised.
            kernel = 2 * int(4.0 * sigma) + 1
            maps = gaussian_blur(maps, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

        return ScoreResult(scores=image_scores, maps=maps.squeeze(1))

    def _reweighted_image_scores(
        self, patches: Tensor, patch_distances: Tensor, nn_index: Tensor
    ) -> Tensor:
        """PatchCore's reweighted image score.

        The base score is the single most anomalous patch, s* = max_p min_m ||p - m||. Taking
        the max rather than the mean is deliberate: a small defect on an otherwise normal
        image would be averaged away.

        s* is then scaled by how isolated its matched bank entry m* is. If m*'s own nearest
        neighbours in the bank are far from the test patch, m* sits in a sparse region of the
        normal distribution -- it is a rare normal example, so matching it is weak evidence of
        normality and the score is kept high. If the whole neighbourhood is equally close, the
        match is well supported and the score is damped.
        """
        assert self.bank is not None
        b = self.cfg.model.n_neighbours
        if b <= 1 or self.bank.shape[0] <= 1:
            return patch_distances.max(dim=1).values

        s_star, p_star = patch_distances.max(dim=1)
        batch_index = torch.arange(patches.shape[0])
        p_star_features = patches[batch_index, p_star]  # (B, D)
        m_star = nn_index[batch_index, p_star]  # (B,)

        neighbour_index = self._bank_neighbours()[m_star]  # (B, b), column 0 is m* itself
        neighbours = self.bank[neighbour_index]  # (B, b, D)
        distances = torch.linalg.vector_norm(
            neighbours - p_star_features.unsqueeze(1), dim=2
        )  # (B, b)

        # The paper writes this as 1 - exp(d_0) / sum_j exp(d_j), which overflows for the
        # distance magnitudes these features produce. That expression is exactly 1 minus the
        # softmax of the distances at position 0, so use softmax and get the shift-invariant
        # stabilisation for free.
        weights = 1.0 - torch.softmax(distances, dim=1)[:, 0]
        return weights * s_star

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        """Write the memory bank and the settings that produced it."""
        if self.bank is None:
            raise RuntimeError("nothing to save; call fit() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"bank": self.bank, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str | Path, cfg: Config, device: str = "cpu") -> PatchCore:
        """Load a memory bank, refusing it if it was built with incompatible settings."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(cfg, device=device)
        model.bank = payload["bank"]
        model.metadata = payload["metadata"]
        model.feature_shape = tuple(model.metadata["feature_shape"])  # type: ignore[assignment]

        # A bank is only meaningful for the exact feature extractor that produced it. Silently
        # scoring new features against an old bank would produce plausible-looking nonsense.
        for key in ("backbone", "layers", "patch_neighbourhood", "image_size", "crop_size"):
            stored = model.metadata[key]
            current = getattr(cfg.model, key)
            if isinstance(stored, list):
                current = list(current)
            if stored != current:
                raise RuntimeError(
                    f"memory bank at {path} was built with {key}={stored!r} but the config "
                    f"says {current!r}; rebuild it with `make train`"
                )
        return model

    @property
    def bank_size_mb(self) -> float:
        """Serialised size of the memory bank in megabytes."""
        if self.bank is None:
            return 0.0
        return self.bank.numel() * self.bank.element_size() / 1e6
