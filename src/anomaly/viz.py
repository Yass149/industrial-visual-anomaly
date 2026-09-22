"""Heatmap overlays. Shared by the evaluation figures and the serving endpoint."""

from __future__ import annotations

import base64
import io

import numpy as np
from matplotlib import colormaps
from PIL import Image
from torch import Tensor


def normalise_map(
    anomaly_map: Tensor | np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Scale an anomaly map to [0, 1].

    vmin/vmax default to the map's own range, which makes a single image legible but means
    two overlays are not comparable to each other -- a perfectly normal image still shows a
    hot region, because its own maximum is mapped to red. Pass explicit bounds (e.g. the
    training-set score range) whenever overlays are shown side by side.
    """
    array = anomaly_map.detach().cpu().numpy() if isinstance(anomaly_map, Tensor) else anomaly_map
    array = array.astype(np.float32)
    lo = float(array.min()) if vmin is None else vmin
    hi = float(array.max()) if vmax is None else vmax
    if hi - lo < 1e-12:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def overlay_heatmap(
    image: Image.Image,
    anomaly_map: Tensor | np.ndarray,
    alpha: float = 0.45,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a colourised anomaly map over an image of the same size."""
    normalised = normalise_map(anomaly_map, vmin, vmax)
    if normalised.shape != (image.height, image.width):
        raise ValueError(
            f"anomaly map {normalised.shape} does not match image "
            f"{(image.height, image.width)}; the overlay would be misaligned"
        )

    coloured = colormaps[colormap](normalised)[..., :3]  # drop alpha channel
    heat = Image.fromarray((coloured * 255).astype(np.uint8))
    return Image.blend(image.convert("RGB"), heat, alpha)


def to_base64_png(image: Image.Image) -> str:
    """Encode an image as a base64 PNG string, for embedding in a JSON response."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
