from collections.abc import Iterable
from enum import Enum
from typing import override

import torch
import torch.nn as tnn
import torchvision.models as models

from utils import TensorShape, compute_shape

__all__ = ["ModelSegment", "SupportedModels"]


class SupportedModels(Enum):
    ALEXNET = "alexnet"
    VGG_11 = "vgg11"
    VGG_13 = "vgg13"
    VGG_16 = "vgg16"
    VGG_19 = "vgg19"
    VGG_11_BN = "vgg11_bn"
    VGG_13_BN = "vgg13_bn"
    VGG_16_BN = "vgg16_bn"
    VGG_19_BN = "vgg19_bn"
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"
    SQUEEZENET_1_0 = "squeezenet1_0"
    SQUEEZENET_1_1 = "squeezenet1_1"
    DENSENET_121 = "densenet121"
    DENSENET_161 = "densenet161"
    DENSENET_169 = "densenet169"
    DENSENET_201 = "densenet201"
    INCEPTION_V3 = "inception_v3"
    GOOGLENET = "googlenet"
    SHUFFLENET_V2_0_5 = "shufflenet_v2_x0_5"
    SHUFFLENET_V2_1_0 = "shufflenet_v2_x1_0"
    SHUFFLENET_V2_1_5 = "shufflenet_v2_x1_5"
    SHUFFLENET_V2_2_0 = "shufflenet_v2_x2_0"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_L = "mobilenet_v3_large"
    MOBILENET_V3_S = "mobilenet_v3_small"
    RESNEXT_50 = "resnext50_32x4d"
    RESNEXT_101_32 = "resnext101_32x8d"
    RESNEXT_101_64 = "resnext101_64x4d"
    WIDERESNET_50_2 = "wide_resnet50_2"
    WIDERESNET_101_2 = "wide_resnet101_2"
    MNASNET_0_5 = "mnasnet0_5"
    MNASNET_0_75 = "mnasnet0_75"
    MNASNET_1_0 = "mnasnet1_0"
    MNASNET_1_3 = "mnasnet1_3"

    def get_modules(self) -> list[tnn.Module]:
        constructor = getattr(models, self.value)
        return list(constructor().children())


class ModelSegment(tnn.Module):
    def __init__(self, model: SupportedModels, index: int | slice) -> None:
        super().__init__()

        as_slice = slice(index) if isinstance(index, int) else index
        modules = model.get_modules()[as_slice]

        self._convolution_layers: tnn.Sequential = tnn.Sequential()
        self._classifier_layers: tnn.Sequential = tnn.Sequential()

        for module in modules:
            self.append(module)

    def compute_shape(
        self, input_shape: TensorShape | None = None
    ) -> TensorShape | int:
        result_shape = 0 if input_shape is None else input_shape

        for module in self.get_modules():
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
                self._classifier_layers.append(module)
            else:
                self._convolution_layers.append(module)
        elif isinstance(module, tnn.Linear):
            self._classifier_layers.append(module)
        else:
            self._convolution_layers.append(module)

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
