"""MVTec AD dataset access and the one deterministic preprocessing path.

Preprocessing lives here and nowhere else. Training, evaluation and the API all call
`build_transform`, so an image scored through the service goes through byte-identical
steps to one scored during evaluation — otherwise the served model is not the evaluated model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ImageNet statistics, because the backbone is frozen ImageNet-pretrained: its features are
# only meaningful for inputs normalised the way it was trained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Split = Literal["train", "test"]


@dataclass(frozen=True)
class Sample:
    """One image with its label, ground-truth mask and provenance."""

    image: torch.Tensor  # (3, crop, crop), normalised
    label: int  # 0 normal, 1 anomalous
    mask: torch.Tensor  # (crop, crop), 1.0 on defective pixels
    path: str
    defect_type: str  # "good" for normal images


def build_transform(image_size: int, crop_size: int) -> transforms.Compose:
    """The image preprocessing pipeline. Deterministic: no augmentation, anywhere.

    PatchCore models the distribution of normal patch features, so augmenting the training
    set would widen that distribution with variation the production camera never produces
    and make the memory bank tolerant of things it should flag.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_display_transform(image_size: int, crop_size: int) -> transforms.Compose:
    """The geometric half of `build_transform`, without normalisation.

    Anomaly maps are produced in the cropped coordinate frame, so an overlay has to be drawn
    on an image that went through the same resize and crop, or the heatmap points at the
    wrong pixels.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop_size),
        ]
    )


def build_mask_transform(image_size: int, crop_size: int) -> transforms.Compose:
    """Mask preprocessing: same geometry as the image, nearest-neighbour to stay binary."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB. Several MVTec categories ship single-channel PNGs."""
    return Image.open(path).convert("RGB")


class MVTecDataset(Dataset[Sample]):
    """One MVTec AD category and split.

    Train split is normal images only, which is the whole point: the model never sees a
    defect. Test split mixes normal images with every defect type the category ships.
    """

    def __init__(
        self,
        root: Path,
        category: str,
        split: Split,
        image_size: int,
        crop_size: int,
    ) -> None:
        self.category_dir = Path(root) / category
        if not self.category_dir.is_dir():
            raise FileNotFoundError(f"category not found: {self.category_dir}")

        self.split = split
        self.crop_size = crop_size
        self.transform = build_transform(image_size, crop_size)
        self.mask_transform = build_mask_transform(image_size, crop_size)

        # Sorted so item order is identical on every machine and every run.
        self.items: list[tuple[Path, int, str]] = []
        for defect_dir in sorted((self.category_dir / split).iterdir()):
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_dir.name == "good" else 1
            for image_path in sorted(defect_dir.glob("*.png")):
                self.items.append((image_path, label, defect_dir.name))

        if not self.items:
            raise FileNotFoundError(f"no images under {self.category_dir / split}")
        if split == "train" and any(label for _, label, _ in self.items):
            raise RuntimeError(f"{category}: train split contains anomalous images")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Sample:
        image_path, label, defect_type = self.items[index]
        image = self.transform(load_image(image_path))

        if label == 0:
            mask = torch.zeros(self.crop_size, self.crop_size)
        else:
            mask_path = (
                self.category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"missing ground-truth mask: {mask_path}")
            mask = self.mask_transform(Image.open(mask_path)).squeeze(0)
            # MVTec masks are 0/255; anything above zero is defective.
            mask = (mask > 0).float()

        return Sample(
            image=image,
            label=label,
            mask=mask,
            path=str(image_path),
            defect_type=defect_type,
        )


def collate_samples(batch: list[Sample]) -> dict[str, object]:
    """Stack a list of Samples into batched tensors, keeping paths and defect types as lists."""
    return {
        "image": torch.stack([s.image for s in batch]),
        "label": torch.tensor([s.label for s in batch], dtype=torch.long),
        "mask": torch.stack([s.mask for s in batch]),
        "path": [s.path for s in batch],
        "defect_type": [s.defect_type for s in batch],
    }
