from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
import torch.nn as tnn

from model_register import load_model_internals, lookup_model
from model_segment import ModelSegment
from tensor_shape import TensorShape


def get_modules(name: str) -> list[tnn.Module]:
    return load_model_internals(lookup_model(name)).modules


def test_init_int_index():
    seg = ModelSegment(get_modules("vgg_16"), index=3)
    assert len(seg.get_modules()) > 0


def test_init_slice():
    seg = ModelSegment(get_modules("squeezenet_1_0"), index=slice(0, 3))
    assert len(seg.get_modules()) > 0


def test_conv_layers_in_convolution_sequential():
    seg = ModelSegment(get_modules("resnet_50"), index=3)
    assert len(seg._convolution_layers) > 0


def test_append_linear_goes_to_classifier():
    seg = ModelSegment(get_modules("alexnet"), index=1)
    seg.append(tnn.Linear(256, 15))
    assert any(isinstance(m, tnn.Linear) for m in seg._classifier_layers)


def test_forward_conv_only():
    seg = ModelSegment(get_modules("alexnet"), index=2)
    x = torch.randn(1, 3, 227, 227)
    out = seg(x)
    assert isinstance(out, torch.Tensor)


def test_compute_shape():
    seg = ModelSegment(get_modules("alexnet"), index=2)
    shape = TensorShape(227, 227, 3)
    result = seg.compute_shape(shape)
    assert isinstance(result, TensorShape)
