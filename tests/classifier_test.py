from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import pytest
import torch.nn as tnn

from classifier import Classifier


def test_linear_rewired_to_input_shape():
    clf = Classifier([tnn.Linear(999, 64)], input_shape=128)
    assert clf._layers[0].in_features == 128
    assert clf._layers[0].out_features == 64


def test_chained_linears_rewired():
    clf = Classifier([tnn.Linear(1, 64), tnn.Linear(1, 32)], input_shape=128)
    assert clf._layers[0].in_features == 128
    assert clf._layers[1].in_features == 64


def test_out_features_tracked():
    clf = Classifier(
        [tnn.Linear(1, 64), tnn.ReLU(), tnn.Linear(1, 15)], input_shape=128
    )
    assert clf.out_features == 15


def test_conv2d_rejected_in_init():
    with pytest.raises(ValueError, match="Convolution layers cannot be in classifier"):
        Classifier([tnn.Conv2d(3, 64, 3)], input_shape=128)


def test_append_conv2d_rejected():
    clf = Classifier([], input_shape=128)
    with pytest.raises(ValueError, match="Convolution layers cannot be in classifier"):
        clf.append(tnn.Conv2d(3, 64, 3))


def test_append_linear_rewired():
    clf = Classifier([tnn.Linear(1, 64)], input_shape=128)
    clf.append(tnn.Linear(1, 32))
    assert clf._layers[-1].in_features == 64
    assert clf.out_features == 32


def test_extend_from_iterable():
    clf = Classifier([tnn.Linear(1, 64)], input_shape=128)
    clf.extend([tnn.ReLU(), tnn.Linear(1, 15)])
    assert clf.out_features == 15


def test_extend_from_classifier():
    clf1 = Classifier([tnn.Linear(1, 64)], input_shape=128)
    clf2 = Classifier([tnn.Linear(1, 15)], input_shape=64)
    clf1.extend(clf2)
    assert clf1.out_features == 15


def test_extend_conv2d_rejected():
    clf = Classifier([], input_shape=128)
    with pytest.raises(ValueError, match="Convolution layers cannot be in classifier"):
        clf.extend([tnn.Conv2d(3, 64, 3)])


def test_sequential_returns_correct_type():
    clf = Classifier([tnn.Linear(1, 64), tnn.ReLU()], input_shape=128)
    seq = clf.sequential()
    assert isinstance(seq, tnn.Sequential)
    assert len(seq) == 2
