"""Landmark image dataset and class label definitions.

This module provides `Marker`, an enum of landmark classes, and `Dataset`,
a `torch.utils.data.Dataset` implementation that loads images from a
directory structure where each subdirectory represents a class.

Expected directory structure::

    images/
        00_OTHER/
            img1.jpg
            img2.jpg
        01_KREMLIN/
            img1.jpg
        ...

Intended usage::

    transform = tt2.Compose([tt2.Resize((227, 227)), tt2.ToTensor()])
    dataset = Dataset("images/", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

from enum import Enum
from os import PathLike
from pathlib import Path
from typing import override

import torch
import torchvision.transforms.v2 as tt2
from torch.utils.data import Dataset as BaseDataset
from torchvision.io import ImageReadMode, decode_image

__all__ = ["Dataset", "Marker"]


class Marker(Enum):
    OTHER = 0
    NN_KREMLIN = 1
    CHKALOV_STAIRCASE = 2
    RUKAVISHNIKOV_ESTATE = 3
    ARHANGELSK_CATHEDRAL = 4
    PECHERSKY_ASCENSION_MONASTERY = 5
    CHURCH_OF_THE_NATIVITY = 6
    STATE_BANK = 7
    PALACE_OF_LABOR = 8
    NN_CATHEDRAL_MOSQUE = 9
    ALEXANDER_NEVSKY_CATHEDRAL = 10
    SPASSKY_OLD_FAIR_CATHEDRAL = 11
    NN_FAIR = 12
    SPB_HERMITAGE = 13
    MOSCOW_KREMLIN = 14
    NN_DRAMA_THEATER_GORKY = 15
    CHURCH_OF_THE_NATIVITY_WITH_THE_ROYAL_CHAPEL = 16
    MOSCOW_ST_BASILS_CATHEDRAL = 17
    MOSCOW_CATHEDRAL_OF_CHRIST_THE_SAVIOR = 18
    SPB_ST_ISAACS_CATHEDRAL = 19
    SPB_RUSSIAN_MUSEUM = 20
    SPB_KAZAN_CATHEDRAL = 21


class Dataset(BaseDataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        images_directory: str | PathLike[str] | Path,
        transform: tt2.Transform | None = None,
        to_float: bool = False,
    ) -> None:
        self._images_directory: Path = Path(images_directory)

        directories = list(self._images_directory.iterdir())
        directories.sort()

        self._pool: list[tuple[Path, Marker]] = self._load_pool(directories)
        self._pool_idx: int = -1

        self._transform: tt2.Compose | tt2.Transform | None = transform
        self.to_float = to_float

    def __len__(self) -> int:
        return len(self._pool)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if self.to_float:
            float_transform = tt2.ConvertImageDtype(torch.float32)
            image = float_transform(
                decode_image(
                    str(image_path), ImageReadMode.RGB, apply_exif_orientation=True
                )
            )
        else:
            image = decode_image(
                str(image_path), ImageReadMode.RGB, apply_exif_orientation=True
            )

        if self._transform is not None:
            image = self._transform(image)

        if not isinstance(image, torch.Tensor):
            raise RuntimeError("Image is not a tensor after transform")

        return image.float()

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self._pool[index]

        return self._load_image(image), label.value

    @staticmethod
    def _load_pool(
        directories: list[Path],
    ) -> list[tuple[Path, Marker]]:
        pool: list[tuple[Path, Marker]] = []

        for directory in directories:
            marker_number = int(directory.name[: directory.name.find("_")])
            marker = Marker(marker_number)

            for photo in directory.iterdir():
                pool.append((photo, marker))

        return pool

    @property
    def pool(self) -> list[tuple[Path, Marker]]:
        return self._pool
