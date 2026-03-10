from collections.abc import Iterable
from enum import Enum, auto
from typing import override

import torch
import torch.nn as tnn
import torchvision.models as models  # pyright: ignore[reportMissingTypeStubs]

from utils import TensorShape, compute_shape


class SupportedModels(Enum):
    ALEXNET = auto


def _get_modules(model: SupportedModels) -> list[tnn.Module]:
    modules: list[tnn.Module] = []

    match model:
        case SupportedModels.ALEXNET:
            modules = list(models.AlexNet().children())

    return modules


def modules_of_model(model: SupportedModels) -> list[tnn.Module]:
    return _get_modules(model)


class ModelSegment(tnn.Module):
    def __init__(self, model: SupportedModels, index: int | slice) -> None:
        super().__init__()
        modules: list[tnn.Module] = []

        if isinstance(index, int):
            modules = _get_modules(model)[0:index]
        else:
            modules = _get_modules(model)[index]

        self._convolution_layers: tnn.Sequential = tnn.Sequential()
        self._classifier_layers: tnn.Sequential = tnn.Sequential()

        for module in modules:
            self.append(module)

    def compute_shape(
        self, input_shape: TensorShape | None = None
    ) -> TensorShape | int:
        result_shape: TensorShape | int = 0 if input_shape is None else input_shape

        for module in self._convolution_layers + self._classifier_layers:
            if isinstance(module, tnn.Linear):
                result_shape = module.out_features
            else:
                result_shape = compute_shape(module, result_shape)

        return result_shape

    def extend(self, modules: Iterable[tnn.Module]) -> None:
        for module in modules:
            self.append(module)

    def append(self, module: tnn.Module) -> None:
        if isinstance(module, tnn.Sequential):
            if any(isinstance(submodule, tnn.Linear) for submodule in module):
                _ = self._classifier_layers.append(module)
            else:
                _ = self._convolution_layers.append(module)
        elif isinstance(module, tnn.Linear):
            _ = self._classifier_layers.append(module)
        else:
            _ = self._convolution_layers.append(module)

    def get_modules(self) -> tnn.Sequential:
        return self._convolution_layers + self._classifier_layers

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._convolution_layers) == 0:
            x = self._classifier_layers(x)  # pyright: ignore[reportAny]
        elif len(self._classifier_layers) == 0:
            x = self._convolution_layers(x)  # pyright: ignore[reportAny]
        else:
            x = self._convolution_layers(x)  # pyright: ignore[reportAny]
            x = torch.flatten(x, 1)
            x = self._classifier_layers(x)  # pyright: ignore[reportAny]

        return x
