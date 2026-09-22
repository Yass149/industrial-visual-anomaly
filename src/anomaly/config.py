"""Typed configuration loaded from a single YAML file.

Every knob in the pipeline lives in configs/default.yaml and is validated here on load.
`extra="forbid"` is deliberate: a mistyped key should fail immediately with the key name
rather than silently leaving a default in place and producing numbers nobody can explain.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 15 categories of the MVTec AD release; used to reject typos in data.categories before a
# download or a training run wastes anyone's time.
MVTEC_CATEGORIES: frozenset[str] = frozenset(
    {
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    }
)

VALID_LAYERS: frozenset[str] = frozenset({"layer1", "layer2", "layer3", "layer4"})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DataConfig(_Base):
    root: Path
    archive_name: str
    url: str
    # None on a fresh clone: the first download prints the digest to paste back in here.
    sha256: str | None = None
    keep_archive: bool = True
    categories: list[str] = Field(min_length=1)

    @field_validator("root")
    @classmethod
    def _expand_root(cls, v: Path) -> Path:
        # The dataset usually lives outside the repo (it is 5GB and often on another volume),
        # so a leading ~ has to work here even though pathlib does not expand it.
        return v.expanduser()

    @field_validator("categories")
    @classmethod
    def _known_categories(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - MVTEC_CATEGORIES)
        if unknown:
            raise ValueError(f"not MVTec AD categories: {unknown}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate categories")
        return v

    @field_validator("sha256")
    @classmethod
    def _looks_like_sha256(cls, v: str | None) -> str | None:
        if v is not None and (len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower())):
            raise ValueError("sha256 must be 64 hex characters")
        return v.lower() if v else v

    @property
    def extract_dir(self) -> Path:
        """Directory holding the per-category trees, e.g. data/mvtec/bottle/train/good."""
        return self.root / "mvtec"

    @property
    def archive_path(self) -> Path:
        return self.root / self.archive_name


class ModelConfig(_Base):
    backbone: Literal["wide_resnet50_2", "resnet50", "resnet18"]
    layers: list[str] = Field(min_length=1)
    image_size: int = Field(gt=0)
    crop_size: int = Field(gt=0)
    patch_neighbourhood: int = Field(gt=0)
    coreset_ratio: float = Field(gt=0.0, le=1.0)
    n_neighbours: int = Field(gt=0)
    gaussian_blur_sigma: float = Field(ge=0.0)

    @field_validator("layers")
    @classmethod
    def _known_layers(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - VALID_LAYERS)
        if unknown:
            raise ValueError(f"unknown backbone layers: {unknown}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate backbone layers")
        return v

    @field_validator("patch_neighbourhood")
    @classmethod
    def _odd_neighbourhood(cls, v: int) -> int:
        # An even window has no centre pixel, so the pooled feature would not be aligned
        # with the patch position it is supposed to describe.
        if v % 2 == 0:
            raise ValueError("patch_neighbourhood must be odd")
        return v

    @model_validator(mode="after")
    def _crop_fits(self) -> ModelConfig:
        if self.crop_size > self.image_size:
            raise ValueError(f"crop_size {self.crop_size} exceeds image_size {self.image_size}")
        return self


class EvalConfig(_Base):
    results_dir: Path
    n_qualitative: int = Field(gt=0)
    calibration_fraction: float = Field(default=0.3, gt=0, lt=1)


class ThresholdConfig(_Base):
    cost_ratio: float = Field(gt=0.0)
    base_rate: float = Field(gt=0.0, lt=1.0)
    cost_ratio_sweep: list[float] = Field(min_length=1)

    @field_validator("cost_ratio_sweep")
    @classmethod
    def _positive_sweep(cls, v: list[float]) -> list[float]:
        if any(r <= 0 for r in v):
            raise ValueError("cost ratios must be positive")
        return sorted(v)


class ServeConfig(_Base):
    host: str
    port: int = Field(gt=0, lt=65536)
    category: str
    artifacts_dir: Path
    max_upload_mb: int = Field(default=10, gt=0)


class MonitoringConfig(_Base):
    drift_alpha: float = Field(gt=0.0, lt=1.0)
    reference_sample: int = Field(gt=0)
    window_size: int = Field(default=64, ge=2)
    projections: int = Field(default=8, ge=1)


class Config(_Base):
    seed: int
    device: Literal["auto", "cpu", "cuda", "mps"]
    data: DataConfig
    model: ModelConfig
    eval: EvalConfig
    threshold: ThresholdConfig
    serve: ServeConfig
    monitoring: MonitoringConfig

    @model_validator(mode="after")
    def _serve_category_is_built(self) -> Config:
        if self.serve.category not in self.data.categories:
            raise ValueError(
                f"serve.category {self.serve.category!r} is not in data.categories "
                f"{self.data.categories}, so no memory bank would exist for it"
            )
        return self


def repo_root() -> Path:
    """Repository root, derived from this file's location rather than the cwd."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate the YAML config, resolving relative paths against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    cfg = Config.model_validate(raw)
    return _absolutise(cfg)


def _absolutise(cfg: Config) -> Config:
    """Make every path absolute so behaviour does not depend on the working directory."""
    root = repo_root()
    updates: dict[str, Any] = {
        "data": cfg.data.model_copy(update={"root": root / cfg.data.root}),
        "eval": cfg.eval.model_copy(update={"results_dir": root / cfg.eval.results_dir}),
        "serve": cfg.serve.model_copy(update={"artifacts_dir": root / cfg.serve.artifacts_dir}),
    }
    return cfg.model_copy(update=updates)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached default config, for callers that do not take a config argument (e.g. the API)."""
    return load_config()


def resolve_device(spec: str) -> str:
    """Turn the config's device spec into a concrete torch device string.

    torch is imported lazily so that config loading stays cheap for callers that never
    touch the model (the download script, the threshold module, config tests).
    """
    import torch

    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"
