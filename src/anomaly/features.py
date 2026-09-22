"""Patch feature extraction from a frozen ImageNet backbone.

Nothing here is trained. The backbone is a fixed function from image to feature maps; all
the "learning" in PatchCore is storing normal patch features and measuring distance to them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

# Backbones with ImageNet weights available in torchvision. WideResNet50-2 is the PatchCore
# paper's choice and the one published MVTec numbers are quoted against; ResNet18 is here so
# the accuracy/latency tradeoff can be measured rather than asserted.
_BACKBONES: dict[str, tuple[object, object]] = {
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


class PatchFeatureExtractor(nn.Module):
    """Extracts a single concatenated patch-feature map from selected backbone layers.

    Produces (B, D, H, W): one D-dimensional descriptor per spatial position, where each
    position corresponds to a patch of the input image.
    """

    def __init__(
        self,
        backbone: str,
        layers: list[str],
        neighbourhood: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unsupported backbone {backbone!r}; known: {sorted(_BACKBONES)}")

        builder, weights = _BACKBONES[backbone]
        self.backbone = builder(weights=weights)  # type: ignore[operator]
        self.layer_names = list(layers)
        self.neighbourhood = neighbourhood
        self.device_str = device

        # Frozen and in eval mode: no gradients, and BatchNorm uses its ImageNet running
        # statistics rather than batch statistics, so a patch's descriptor does not depend
        # on which other images happened to be in its batch.
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self._captured: dict[str, Tensor] = {}
        for name in self.layer_names:
            module = getattr(self.backbone, name, None)
            if module is None:
                raise ValueError(f"backbone {backbone!r} has no layer {name!r}")
            module.register_forward_hook(self._make_hook(name))

        self.backbone.to(device)

    def _make_hook(self, name: str):  # noqa: ANN202 - returns an internal closure
        def hook(_module: nn.Module, _inputs: object, output: Tensor) -> None:
            self._captured[name] = output

        return hook

    @property
    def feature_dim(self) -> int:
        """Descriptor length D, i.e. the summed channel counts of the selected layers."""
        with torch.no_grad():
            probe = torch.zeros(1, 3, 64, 64, device=self.device_str)
            return int(self(probe).shape[1])

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Map a batch of images to patch features of shape (B, D, H, W)."""
        self._captured.clear()
        self.backbone(images.to(self.device_str))
        maps = [self._captured[name] for name in self.layer_names]

        # Average over a local neighbourhood so each descriptor summarises its surroundings
        # instead of one receptive field. This is what makes a patch feature robust enough
        # that a nearest-neighbour lookup means something.
        pad = self.neighbourhood // 2
        maps = [F.avg_pool2d(m, self.neighbourhood, stride=1, padding=pad) for m in maps]

        # Deeper layers are spatially coarser. Upsample them to the first layer's resolution
        # so descriptors from different layers describe the same patch before concatenation.
        target_size = maps[0].shape[-2:]
        maps = [
            (
                m
                if m.shape[-2:] == target_size
                else F.interpolate(m, size=target_size, mode="bilinear")
            )
            for m in maps
        ]
        return torch.cat(maps, dim=1)


def flatten_patches(features: Tensor) -> Tensor:
    """(B, D, H, W) -> (B*H*W, D), one row per patch, row-major in (b, h, w)."""
    batch, dim, height, width = features.shape
    return features.permute(0, 2, 3, 1).reshape(batch * height * width, dim)
