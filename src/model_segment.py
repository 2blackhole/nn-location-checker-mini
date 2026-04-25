"""Model registry and segmentation for feature extraction.

This module provides `SupportedModels`, an enum of torchvision
models, and `ModelSegment`, which slices a donor into a reusable
`torch.nn.Module` separating convolutional and classifier layers.
"""

from collections.abc import Iterable
from typing import cast, override

import torch
import torch.nn as tnn
from timm.models import FastVit, VisionTransformer

from tensor_shape import TensorShape, compute_shape


class ModelSegment(tnn.Module):
    def __init__(
        self,
        modules: list[tnn.Module],
        index: int | slice,
    ) -> None:
        super().__init__()

        as_slice = slice(index) if isinstance(index, int) else index
        selected_modules = modules[as_slice]

        self._convolution_layers: tnn.Sequential = tnn.Sequential()
        self._classifier_layers: tnn.Sequential = tnn.Sequential()

        for module in selected_modules:
            self.append(module)

    def compute_shape(self, input_shape: TensorShape) -> TensorShape | int:
        result_shape = input_shape
        for module in self.get_modules():
            if isinstance(module, tnn.Linear):
                result_shape = module.out_features
            elif (
                isinstance(module, tnn.Sequential)
                and len(list(module.children())) > 0
                and isinstance(module[-1], tnn.Linear)
            ):
                result_shape = module[-1].out_features
            elif isinstance(module, FastVit):
                result_shape = module.head.fc.out_features
            elif isinstance(module, VisionTransformer):
                result_shape = cast(int, module.head.out_features)
            else:
                result_shape = compute_shape(module, result_shape)

        return result_shape

    def extend(self, modules: Iterable[tnn.Module]) -> None:
        for module in modules:
            self.append(module)

    def append(self, module: tnn.Module) -> None:
        if isinstance(module, tnn.Sequential):
            if any(isinstance(submodule, tnn.Linear) for submodule in module):
                self._classifier_layers.append(module)
            else:
                self._convolution_layers.append(module)
        elif isinstance(module, tnn.Linear):
            self._classifier_layers.append(module)
        else:
            self._convolution_layers.append(
                module
            )  # For the future: _DenseBlock & _Transition come here too

    def get_modules(self) -> tnn.Sequential:
        return self._convolution_layers + self._classifier_layers

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._convolution_layers) == 0:
            x = self._classifier_layers(x)
        elif len(self._classifier_layers) == 0:
            x = self._convolution_layers(x)
        else:
            x = self._convolution_layers(x)
            x = torch.flatten(x, 1)
            x = self._classifier_layers(x)

        return x