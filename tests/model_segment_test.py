from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
import torch.nn as tnn

from model_segment import ModelSegment, SupportedModels
from utils import TensorShape


def test_init_int_index():
    seg = ModelSegment(SupportedModels.VGG_16, index=3)
    assert len(seg.get_modules()) > 0


def test_init_slice():
    seg = ModelSegment(SupportedModels.SQUEEZENET_1_0, index=slice(0, 3))
    assert len(seg.get_modules()) > 0


def test_conv_layers_in_convolution_sequential():
    seg = ModelSegment(SupportedModels.RESNET_50, index=3)
    assert len(seg._convolution_layers) > 0


def test_append_linear_goes_to_classifier():
    seg = ModelSegment(SupportedModels.ALEXNET, index=1)
    seg.append(tnn.Linear(256, 15))
    assert any(isinstance(m, tnn.Linear) for m in seg._classifier_layers)


def test_forward_conv_only():
    seg = ModelSegment(SupportedModels.ALEXNET, index=2)
    x = torch.randn(1, 3, 227, 227)
    out = seg(x)
    assert isinstance(out, torch.Tensor)


def test_compute_shape():
    seg = ModelSegment(SupportedModels.ALEXNET, index=2)
    shape = TensorShape(227, 227, 3)
    result = seg.compute_shape(shape)
    assert isinstance(result, TensorShape)
