import json
from pathlib import Path

import torch.nn as tnn

from classifier import Classifier
from utils import TensorShape


def _build_conv2d(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    out = dct.get("out")
    match out:
        case (int(), int()):
            raise ValueError("UB")
        case None:
            raise ValueError("There must be out channels value")
        case _:
            out = int(out)
            pass

    kernel = dct.get("kernel")
    match kernel:
        case (int(), int()):
            raise ValueError("UB")
        case None:
            raise ValueError("There must be kernel size value")
        case _:
            kernel = int(kernel)
            pass

    stride = dct.get("stride")
    match stride:
        case (int(), int()):
            raise ValueError("UB")
        case None:
            stride = 1
            pass
        case _:
            stride = int(stride)
            pass

    padding = dct.get("padding")
    match padding:
        case (int(), int()):
            raise ValueError("UB")
        case None:
            padding = 0
            pass
        case _:
            padding = int(padding)
            pass

    return tnn.Conv2d(1, out, kernel, stride, padding)


def _build_activation(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    function_type = dct.get("function")
    if function_type is None:
        raise ValueError("There must be activation function type")
    else:
        function_type = str(function_type)

    inplace = bool(dct.get("inplace", False))

    match function_type:
        case "relu":
            return tnn.ReLU(inplace)
        case _:
            raise NotImplementedError()


def _build_pool(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    pool_type = dct.get("function")
    if pool_type is None:
        raise ValueError("There must be pool function type")
    else:
        pool_type = str(pool_type)

    kernel = dct.get("kernel")
    match kernel:
        case (int(), int()):
            raise ValueError("UB")
        case None:
            raise ValueError("There must be kernel size value")
        case _:
            kernel = int(kernel)
            pass

    stride = dct.get("stride")
    match stride:
        case (int(), int()):
            raise ValueError("UB")
        case None:
            raise ValueError("There must be kernel size value")
        case _:
            stride = int(stride)

    match pool_type:
        case "max":
            return tnn.MaxPool2d(kernel, stride)
        case "avg":
            return tnn.AvgPool2d(kernel, stride)
        case _:
            raise NotImplementedError()


def _build_adaptive_pool(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    pool_type = dct.get("function")
    if pool_type is None:
        raise ValueError("There must be pool function type")
    else:
        pool_type = str(pool_type)

    out_size = dct.get("out")
    match out_size:
        case None:
            raise ValueError("There must be out size value")
        case (int(), int()):
            out_size = tuple(out_size)
        case _:
            raise ValueError("UB")

    match pool_type:
        case "max":
            return tnn.AdaptiveMaxPool2d(out_size)
        case "avg":
            return tnn.AdaptiveAvgPool2d(out_size)
        case _:
            raise ValueError("UB")


def _build_dropout(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    percent = dct.get("percent")
    match percent:
        case None:
            raise ValueError("There must be percent value")
        case (int(), int()):
            raise ValueError("UB")
        case _:
            percent = float(percent)

    inplace = bool(dct.get("inplace", False))

    return tnn.Dropout(percent, inplace)


def _build_linear(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    out = dct.get("out")
    match out:
        case None:
            raise ValueError("There must be out features value")
        case (int(), int()):
            raise ValueError("UBJ")
        case _:
            out = int(out)

    bias = bool(dct.get("bias", True))

    return tnn.Linear(1, out, bias)


def _as_module_data(dct: dict[str, str | int | bool | tuple[int, int] | float]):
    module_type = dct.get("type")
    if module_type is None:
        raise ValueError("There must be module type")
    else:
        module_type = str(module_type)
    match module_type:
        case "convolution":
            return _build_conv2d(dct)
        case "activation":
            return _build_activation(dct)
        case "pool":
            return _build_pool(dct)
        case "adaptive_pool":
            return _build_adaptive_pool(dct)
        case "dropout":
            return _build_dropout(dct)
        case "linear":
            return _build_linear(dct)
        case _:
            raise ValueError("""Field "type" is incorrect""")


class ModuleLoader:
    def __init__(self, file: Path | str) -> None:
        self._file: Path = Path(file)
        self._modules: list[tnn.Module] = []
        if not self._file.exists():
            raise ValueError("File does not exist")

        with open(self._file, encoding="utf-8") as module_file:
            data = json.load(module_file, object_hook=_as_module_data)  # pyright: ignore[reportAny]

        if not isinstance(data, list):
            raise ValueError("Path leads to incorrect module file")

        for record in data:  # pyright: ignore[reportUnknownVariableType]
            self._modules.append(record)  # pyright: ignore[reportUnknownArgumentType]

    def load(self, input_shape: TensorShape) -> Classifier:
        result: Classifier | None = None
        result = Classifier(self._modules, input_shape)

        return result
