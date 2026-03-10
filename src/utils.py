from functools import singledispatch
from typing import NamedTuple, Never, overload

import torch.nn as tnn


class TensorShape(NamedTuple):
    height: int
    width: int
    channels: int

    def in_features(self) -> int:
        return self.height * self.channels * self.width


@overload
def _to_pair(obj: int | tuple[int, int]) -> tuple[int, int]: ...


@overload
def _to_pair(obj: tuple[int | None, int | None]) -> tuple[int | None, int | None]: ...


@overload
def _to_pair(obj: None) -> Never: ...


def _to_pair(obj: int | tuple[int, int] | tuple[int | None, int | None] | None):
    match obj:
        case int():
            return (obj, obj)
        case None:
            raise ValueError("Shape does not exist")
        case _:
            return obj


@singledispatch
def compute_shape(module: tnn.Module, previous_shape: TensorShape) -> TensorShape:  # pyright: ignore[reportUnusedParameter]
    raise NotImplementedError(
        f"Cannot compute features map for the module: {type(module).__name__}"
    )


def compute_conv(origin: int, padding: int, kernel_size: int, stride: int) -> int:
    return int((origin + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


@compute_shape.register
def _(module: tnn.Conv2d, previous_shape: TensorShape) -> TensorShape:
    if isinstance(module.padding, str):
        raise NotImplementedError("Padding is string")
    new_height = compute_conv(
        previous_shape.height,
        module.padding[0],
        module.kernel_size[0],
        module.stride[0],
    )
    new_width = compute_conv(
        previous_shape.width, module.padding[1], module.kernel_size[1], module.stride[1]
    )
    return TensorShape(new_height, new_width, module.out_channels)


@compute_shape.register
def _(
    _module: tnn.ReLU | tnn.Dropout | tnn.BatchNorm2d, previous_shape: TensorShape
) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(module: tnn.MaxPool2d, previous_shape: TensorShape) -> TensorShape:
    padding_height, padding_width = _to_pair(module.padding)
    kernel_height, kernel_width = _to_pair(module.kernel_size)
    stride_height, stride_width = _to_pair(module.stride)

    new_height = compute_conv(
        previous_shape.height, padding_height, kernel_height, stride_height
    )
    new_width = compute_conv(
        previous_shape.width, padding_width, kernel_width, stride_width
    )
    return TensorShape(new_height, new_width, previous_shape.channels)


@compute_shape.register
def _(
    module: tnn.AdaptiveAvgPool2d | tnn.AdaptiveMaxPool2d, previous_shape: TensorShape
) -> TensorShape:
    new_height, new_width = _to_pair(module.output_size)

    if new_height is None:
        new_height = previous_shape.height

    if new_width is None:
        new_width = previous_shape.width

    return TensorShape(
        new_height,
        new_width,
        previous_shape.channels,
    )


@compute_shape.register
def _(module: tnn.Sequential, previous_shape: TensorShape) -> TensorShape:
    result_shape: TensorShape = previous_shape
    for submodule in module:
        result_shape = compute_shape(submodule, result_shape)

    return result_shape
