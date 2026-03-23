"""Linear classifier builder for image classification pipelines.

This module provides :class:`Classifier`, which wraps a sequence of
:class:`torch.nn.Module` layers into a classifier,
automatically rewiring :class:`~torch.nn.Linear` layers to match
the input dimension.

"""

from collections.abc import Iterable, Sequence
from typing import Self, cast, overload

import torch.nn as tnn

from utils import TensorShape

__all__ = ["Classifier"]


def _reject_conv2d(modules: Iterable[tnn.Module]) -> None:
    if any(isinstance(m, tnn.Conv2d) for m in modules):
        raise ValueError("Convolution layers cannot be in classifier")


class Classifier:
    def __init__(
        self, modules: Sequence[tnn.Module], input_shape: TensorShape | int
    ) -> None:
        _reject_conv2d(modules)

        in_features = (
            input_shape.in_features()
            if isinstance(input_shape, TensorShape)
            else input_shape
        )

        self._layers: list[tnn.Module] = list(modules)
        previous_features: int = in_features

        for i, module in enumerate(modules):
            if isinstance(module, tnn.Linear):
                self._layers[i] = tnn.Linear(previous_features, module.out_features)
                previous_features = module.out_features

        self._out_features = previous_features

    def __repr__(self) -> str:
        return str(self._layers)

    def append(self, module: tnn.Module) -> None:
        if isinstance(module, tnn.Conv2d):
            raise ValueError("Convolution layers cannot be in classifier")

        if isinstance(module, tnn.Linear):
            tail = tnn.Linear(self._out_features, module.out_features)
        else:
            tail = module

        self._layers.append(tail)
        self._out_features = (
            tail.out_features if isinstance(tail, tnn.Linear) else self._out_features
        )

    @overload
    def extend(self, modules: Self) -> None: ...

    @overload
    def extend(self, modules: Iterable[tnn.Module]) -> None: ...

    def extend(self, modules: Iterable[tnn.Module] | Self) -> None:
        if isinstance(modules, self.__class__):
            for module in modules._layers:
                self.append(module)
        else:
            layers = cast(Iterable[tnn.Module], modules)
            layers = list(layers)
            _reject_conv2d(layers)
            for module in layers:
                self.append(module)

    @property
    def out_features(self) -> int:
        return self._out_features

    def sequential(self) -> tnn.Sequential:
        return tnn.Sequential(*self._layers)
