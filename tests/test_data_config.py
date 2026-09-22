from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from anomaly.config import Config, load_config
from anomaly.data import MVTecDataset, build_display_transform, build_mask_transform


def test_config_rejects_typos_and_nonfinite_costs():
    raw = load_config().model_dump()
    raw["threshold"]["cost_ratio"] = float("inf")
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
    raw = load_config().model_dump()
    raw["model"]["layers"] = ["layer2", "layer2"]
    with pytest.raises(ValidationError, match="duplicate"):
        Config.model_validate(raw)
    raw = load_config().model_dump()
    raw["model"]["crop_szie"] = 224
    with pytest.raises(ValidationError, match="crop_szie"):
        Config.model_validate(raw)


def write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_train_rejects_anomalous_folders(tmp_path):
    write_image(tmp_path / "bottle/train/crack/000.png", np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="anomalous"):
        MVTecDataset(tmp_path, "bottle", "train", 16, 16)


def test_masks_missing_then_correctly_binary_and_aligned(tmp_path):
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[8:16, 16:24] = 255
    write_image(tmp_path / "bottle/test/crack/000.png", np.repeat(pixels[:, :, None], 3, axis=2))
    dataset = MVTecDataset(tmp_path, "bottle", "test", 32, 24)
    with pytest.raises(FileNotFoundError, match="mask"):
        dataset[0]
    write_image(tmp_path / "bottle/ground_truth/crack/000_mask.png", pixels)
    sample = dataset[0]
    display = np.asarray(build_display_transform(32, 24)(Image.fromarray(pixels))) > 0
    assert np.array_equal(display, sample.mask.numpy().astype(bool))
    assert set(sample.mask.unique().tolist()) == {0.0, 1.0}
    resized = build_mask_transform(27, 24)(Image.fromarray(pixels))
    assert set(resized.unique().tolist()) == {0.0, 1.0}
