import json
from pathlib import Path
from typing import Any, cast

import torch.nn as tnn

from classifier import Classifier
from utils import TensorShape


def _require_int(dct: dict[str, Any], key: str) -> int:
    val = dct.get(key)
    if val is None:
        raise ValueError(f"Missing required field: '{key}'")
    if isinstance(val, tuple):
        raise ValueError(f"Field '{key}' must be a scalar, not a tuple")
    return int(val)


def _optional_int(dct: dict[str, Any], key: str, default: int) -> int:
    val = dct.get(key)
    if val is None:
        return default
    if isinstance(val, tuple):
        raise ValueError(f"Field '{key}' must be a scalar, not a tuple")
    return int(val)


def _require_float(dct: dict[str, Any], key: str) -> float:
    val = dct.get(key)
    if val is None:
        raise ValueError(f"Missing required field: '{key}'")
    if isinstance(val, tuple):
        raise ValueError(f"Field '{key}' must be a scalar, not a tuple")
    return float(val)


def _require_str(dct: dict[str, Any], key: str) -> str:
    val = dct.get(key)
    if val is None:
        raise ValueError(f"Missing required field: '{key}'")
    return str(val)


def _build_conv2d(
    dct: dict[str, Any],
) -> tnn.Conv2d:
    return tnn.Conv2d(
        1,
        _require_int(dct, "out"),
        _require_int(dct, "kernel"),
        _optional_int(dct, "stride", 1),
        _optional_int(dct, "padding", 0),
    )


def _build_activation(
    dct: dict[str, Any],
) -> tnn.ReLU:
    match _require_str(dct, "function"):
        case "relu":
            return tnn.ReLU(bool(dct.get("inplace", False)))
        case fn:
            raise NotImplementedError(f"Activation '{fn}' not supported")


def _build_pool(
    dct: dict[str, Any],
) -> tnn.MaxPool2d | tnn.AvgPool2d:
    kernel = _require_int(dct, "kernel")
    stride = _require_int(dct, "stride")
    match _require_str(dct, "function"):
        case "max":
            return tnn.MaxPool2d(kernel, stride)
        case "avg":
            return tnn.AvgPool2d(kernel, stride)
        case fn:
            raise NotImplementedError(f"Pool '{fn}' not supported")


def _build_adaptive_pool(
    dct: dict[str, Any],
) -> tnn.AdaptiveAvgPool2d | tnn.AdaptiveMaxPool2d:
    out_size = dct.get("out")
    if not isinstance(out_size, list) or len(out_size) != 2:
        raise ValueError("'out' must be a (int, int) tuple for adaptive pool")
    match _require_str(dct, "function"):
        case "max":
            return tnn.AdaptiveMaxPool2d(out_size)
        case "avg":
            return tnn.AdaptiveAvgPool2d(out_size)
        case fn:
            raise ValueError(f"Unknown adaptive pool type: '{fn}'")


def _build_dropout(
    dct: dict[str, Any],
) -> tnn.Dropout:
    return tnn.Dropout(_require_float(dct, "percent"), bool(dct.get("inplace", False)))


def _build_linear(
    dct: dict[str, Any],
) -> tnn.Linear:
    return tnn.Linear(1, _require_int(dct, "out"), bool(dct.get("bias", True)))


_BUILDERS = {
    "convolution": _build_conv2d,
    "activation": _build_activation,
    "pool": _build_pool,
    "adaptive_pool": _build_adaptive_pool,
    "dropout": _build_dropout,
    "linear": _build_linear,
}


def _as_module_data(
    dct: dict[str, str | int | bool | tuple[int, int] | float],
) -> (
    tnn.Conv2d
    | tnn.ReLU
    | tnn.AdaptiveAvgPool2d
    | tnn.AdaptiveMaxPool2d
    | tnn.MaxPool2d
    | tnn.AvgPool2d
    | tnn.Dropout
    | tnn.Linear
):
    module_type = _require_str(dct, "type")
    builder = _BUILDERS.get(module_type)
    if builder is None:
        raise ValueError(f"Unknown module type: '{module_type}'")
    return builder(dct)


class ModuleLoader:
    def __init__(self, file: Path | str) -> None:
        self._file: Path = Path(file)
        self._modules: list[tnn.Module] = []
        if not self._file.exists():
            raise ValueError("File does not exist")

        with self._file.open(encoding="utf-8") as module_file:
            data = json.load(module_file, object_hook=_as_module_data)

        if not isinstance(data, list):
            raise ValueError("JSON root object must be an array of layers")

        self._modules = cast(list[tnn.Module], list(data))

    def load(self, input_shape: TensorShape | int) -> Classifier:
        return Classifier(self._modules, input_shape)
