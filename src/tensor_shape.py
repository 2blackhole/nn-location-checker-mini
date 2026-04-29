"""Tensor shape and computation for PyTorch modules.

This module provides `TensorShape` for representing tensor
dimensions and `compute_shape` for passing shapes through
`torch.nn.Module` layers without running a forward pass.
"""

from functools import singledispatch
from typing import NamedTuple, Never, overload

import torch.nn as tnn
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition
from torchvision.models.googlenet import BasicConv2d as GooglenetBasicConv2d
from torchvision.models.googlenet import Inception
from torchvision.models.inception import BasicConv2d as InceptionBasicConv2d
from torchvision.models.mnasnet import _InvertedResidual
from torchvision.models.mobilenetv2 import (
    InvertedResidual as Mobilenetv2InvertedResidual,
)
from torchvision.models.mobilenetv3 import (
    InvertedResidual as Mobilenetv3InvertedResidual,
)
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.shufflenetv2 import (
    InvertedResidual as ShufflenetInvertedResidual,
)
from torchvision.models.squeezenet import Fire

__all__ = ["TensorShape", "compute_conv", "compute_shape"]


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


@overload
def compute_shape(module: tnn.Conv2d, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(
    _module: tnn.ReLU | tnn.Dropout | tnn.BatchNorm2d | tnn.ReLU6 | tnn.Hardswish,
    previous_shape: TensorShape,
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.MaxPool2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.AdaptiveAvgPool2d | tnn.AdaptiveMaxPool2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.Sequential, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(module: _DenseLayer, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(module: _DenseBlock, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(module: _Transition, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.AvgPool2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(module: BasicBlock, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(module: Bottleneck, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(module: Fire, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(
    module: Mobilenetv2InvertedResidual, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: Mobilenetv3InvertedResidual, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: InceptionBasicConv2d | GooglenetBasicConv2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(module: Inception, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(
    module: ShufflenetInvertedResidual, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(module: tnn.Module, previous_shape: TensorShape) -> Never: ...


def compute_conv(
    origin: int, padding: int, kernel_size: int, stride: int, dilation: int
) -> int:
    return (origin + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@singledispatch
def compute_shape(module: tnn.Module, _previous_shape: TensorShape) -> Never:
    raise NotImplementedError(
        f"Cannot compute features map for the module: {type(module).__name__}"
    )


@compute_shape.register
def _(module: tnn.Conv2d, previous_shape: TensorShape) -> TensorShape:
    if isinstance(module.padding, str):
        raise NotImplementedError("Padding is string")
    new_height = compute_conv(
        previous_shape.height,
        module.padding[0],
        module.kernel_size[0],
        module.stride[0],
        module.dilation[0],
    )
    new_width = compute_conv(
        previous_shape.width,
        module.padding[1],
        module.kernel_size[1],
        module.stride[1],
        module.dilation[1],
    )
    return TensorShape(new_height, new_width, module.out_channels)


@compute_shape.register
def _(
    _module: tnn.ReLU | tnn.Dropout | tnn.BatchNorm2d | tnn.ReLU6 | tnn.Hardswish,
    previous_shape: TensorShape,
) -> TensorShape:
    return previous_shape


def compute_pool(
    origin: int, padding: int, kernel: int, stride: int, dilation: int, ceil_mode: bool
) -> int:
    dimension = origin + 2 * padding - dilation * (kernel - 1) - 1 + stride
    if ceil_mode:
        return (
            dimension + stride - 1
        ) // stride  # (x + y - 1) // y === ceil(x / y) for x > 0, y > 0
    return dimension // stride


@compute_shape.register
def _(module: tnn.MaxPool2d, previous_shape: TensorShape) -> TensorShape:
    padding_height, padding_width = _to_pair(module.padding)
    kernel_height, kernel_width = _to_pair(module.kernel_size)
    stride_height, stride_width = _to_pair(module.stride)
    dilation_height, dilation_width = _to_pair(module.dilation)

    new_height = compute_pool(
        previous_shape.height,
        padding_height,
        kernel_height,
        stride_height,
        dilation_height,
        module.ceil_mode,
    )
    new_width = compute_pool(
        previous_shape.width,
        padding_width,
        kernel_width,
        stride_width,
        dilation_width,
        module.ceil_mode,
    )
    return TensorShape(new_height, new_width, previous_shape.channels)


@compute_shape.register
def _(module: tnn.AvgPool2d, previous_shape: TensorShape) -> TensorShape:
    padding_height, padding_width = _to_pair(module.padding)
    kernel_height, kernel_width = _to_pair(module.kernel_size)
    stride_height, stride_width = _to_pair(module.stride)

    new_height = compute_pool(
        previous_shape.height,
        padding_height,
        kernel_height,
        stride_height,
        1,
        module.ceil_mode,
    )
    new_width = compute_pool(
        previous_shape.width,
        padding_width,
        kernel_width,
        stride_width,
        1,
        module.ceil_mode,
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
def _(
    module: tnn.Sequential,
    previous_shape: TensorShape,
) -> TensorShape:
    result_shape = previous_shape
    for submodule in module:
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: _DenseLayer, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return TensorShape(
        result_shape.height,
        result_shape.width,
        previous_shape.channels + result_shape.channels,
    )


@compute_shape.register
def _(module: _DenseBlock, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for layer in module.children():
        result_shape = compute_shape(layer, result_shape)
    return result_shape


@compute_shape.register
def _(module: _Transition, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: BasicBlock, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for name, submodule in module.named_children():
        if name == "downsample":
            continue
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: Bottleneck, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for name, submodule in module.named_children():
        if name == "downsample":
            continue
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: Fire, previous_shape: TensorShape) -> TensorShape:
    out_channels = module.expand1x1.out_channels + module.expand3x3.out_channels
    return TensorShape(previous_shape.height, previous_shape.width, out_channels)


@compute_shape.register
def _(module: _InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    stride = next(
        m.stride[0]
        for m in module.layers.modules()
        if isinstance(m, tnn.Conv2d) and m.groups > 1
    )
    out_channels = next(
        m.out_channels
        for m in reversed(list(module.layers.modules()))
        if isinstance(m, tnn.Conv2d)
    )
    return TensorShape(
        previous_shape.height // stride, previous_shape.width // stride, out_channels
    )


@compute_shape.register
def _(module: Mobilenetv2InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    channels = next(
        submodule.out_channels
        for submodule in reversed(list(module.conv))
        if isinstance(submodule, tnn.Conv2d)
    )
    return TensorShape(
        previous_shape.height // module.stride,
        previous_shape.height // module.stride,
        channels,
    )


@compute_shape.register
def _(module: Mobilenetv3InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    stride = next(
        submodule.stride[0]
        for submodule in module.block.modules()
        if isinstance(submodule, tnn.Conv2d) and submodule.groups > 1
    )
    return TensorShape(
        previous_shape.height // stride,
        previous_shape.width // stride,
        module.out_channels,
    )


@compute_shape.register
def _(
    module: InceptionBasicConv2d | GooglenetBasicConv2d, previous_shape: TensorShape
) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: Inception, previous_shape: TensorShape) -> TensorShape:
    branch1_shape = compute_shape(module.branch1, previous_shape)
    branch2_shape = compute_shape(module.branch2, previous_shape)
    branch3_shape = compute_shape(module.branch3, previous_shape)
    branch4_shape = compute_shape(module.branch4, previous_shape)
    return TensorShape(
        previous_shape.height,
        previous_shape.width,
        branch1_shape.channels
        + branch2_shape.channels
        + branch3_shape.channels
        + branch4_shape.channels,
    )


def _branch_channels(branch: tnn.Sequential) -> int:
    for module in reversed(branch):
        if isinstance(module, tnn.Conv2d):
            return module.out_channels
    raise ValueError("No Conv2d was detected in branch")


@compute_shape.register
def _(module: ShufflenetInvertedResidual, previous_shape: TensorShape) -> TensorShape:
    if module.stride == 1:
        return previous_shape
    channels = _branch_channels(module.branch1) + _branch_channels(module.branch2)
    return TensorShape(previous_shape.height // 2, previous_shape.width // 2, channels)
