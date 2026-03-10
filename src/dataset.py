from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Self, overload, override

import torch
import torchvision.transforms.v2 as tt2  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import Dataset as BaseDataset

# isort: off
from torchvision.io import decode_image  # pyright: ignore[reportMissingTypeStubs]
# isort: on


class Marker(Enum):
    OTHER = 0
    KREMLIN = 1
    CHKALOV_STAIRCASE = 2
    RUKAVISHNIKOV_ESTATE = 3
    ARHANGELSK_CATHEDRAL = 4
    PECHERSKY_MONASTERY = 5
    CHURCH_OF_THE_NATIVITY = 6
    STATE_BANK = 7
    PALACE_OF_LABOR = 8
    CATHEDRAL_MOSQUE = 9
    ALEXANDER_NEVSKY_CATHEDRAL = 10
    SPASSKY_OLD_FAIR_CATHEDRAL = 11
    FAIR = 12
    DRAMA_THEATER_GORKY = 13
    CHURCH_OF_THE_NATIVITY_WITH_THE_ROYAL_CHAPEL = 14


class Dataset(BaseDataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        images_directory: str | PathLike[str] | Path,
        transform: tt2.Compose | tt2.Transform | None = None,
    ) -> None:
        self._images_directory: Path = Path(images_directory)

        directories = list(self._images_directory.iterdir())
        directories.sort()

        self._pool: list[tuple[Path, Marker]] = self._load_pool(directories)
        self._pool_idx: int = -1

        self._transform: tt2.Compose | tt2.Transform | None = transform

    def __len__(self) -> int:
        return len(self._pool)

    @overload
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]: ...

    @overload
    def __getitem__(self, idx: slice) -> list[tuple[torch.Tensor, int]]: ...

    @override
    def __getitem__(
        self, idx: int | slice
    ) -> tuple[torch.Tensor, int] | list[tuple[torch.Tensor, int]]:
        def load_image(image_path: Path) -> torch.Tensor:
            image = decode_image(str(image_path), apply_exif_orientation=True)

            if self._transform is not None:
                image = self._transform(image)  # pyright: ignore[reportAny]

            if not isinstance(image, torch.Tensor):
                raise RuntimeError("Image is not a tensor after transform")

            return image.float()

        if isinstance(idx, int):
            image_path, label = self._pool[idx]

            image = load_image(image_path)

            return image, label.value
        else:
            pool_slice = self._pool[idx]

            result_slice: list[tuple[torch.Tensor, int]] = []
            for image_path, label in pool_slice:
                image = load_image(image_path)
                result_slice.append((image, label.value))

            return result_slice

    def __iter__(self) -> Self:
        self._pool_idx = -1
        return self

    def __next__(self) -> tuple[torch.Tensor, int]:
        self._pool_idx += 1

        if self._pool_idx > len(self._pool):
            raise StopIteration

        return self[self._pool_idx]

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
