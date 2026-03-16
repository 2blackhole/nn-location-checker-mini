from collections.abc import Iterable
from functools import singledispatch
from typing import Self

import torch.nn as tnn

from utils import TensorShape


class Classifier:
    def __init__(
        self, modules: list[tnn.Module], input_shape: TensorShape | int
    ) -> None:
        if any(isinstance(module, tnn.Conv2d) for module in modules):
            raise ValueError("Convolution layers can not be in classifier")

        in_features = (
            input_shape.in_features()
            if isinstance(input_shape, TensorShape)
            else input_shape
        )
        self._modules: list[tnn.Module] = modules
        previous_features: int = in_features

        for i, module in enumerate(modules):
            if isinstance(module, tnn.Linear):
                self._modules[i] = tnn.Linear(previous_features, module.out_features)
                previous_features = module.out_features

        self._out_features = previous_features

    def append(self, module: tnn.Module) -> None:
        if isinstance(module, tnn.Conv2d):
            raise ValueError("Convolution layers can not be in classifier")

        if isinstance(module, tnn.Linear):
            tail = tnn.Linear(self._out_features, module.out_features)
        else:
            tail = module

        self._modules.append(tail)
        self._out_features = (
            tail.out_features if isinstance(tail, tnn.Linear) else self._out_features
        )

    @singledispatch
    def extend(self, arg: Iterable[tnn.Module] | Self) -> None:
        if isinstance(arg, self.__class__):
            for module in arg._modules:
                self.append(module)
        else:
            raise NotImplementedError(
                f"Unsupported argument type: {type(arg).__name__}"
            )

    @extend.register(Iterable)
    def _(self, modules: Iterable[tnn.Module]) -> None:
        if any(isinstance(module, tnn.Conv2d) for module in modules):
            raise ValueError("Convolution layers can not be in classifier")

        for module in modules:
            self.append(module)

    @property
    def out_features(self) -> int:
        return self._out_features

    def sequential(self) -> tnn.Sequential:
        return tnn.Sequential(*self._modules)
