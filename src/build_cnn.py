from typing import override

import torch
import torch.nn as tnn

from classifier import Classifier
from model_segment import ModelSegment


class CNnetwork(tnn.Module):
    def __init__(self, model_part: ModelSegment, classifier: Classifier) -> None:
        super().__init__()

        self._model_part: ModelSegment = model_part
        self._classifier: tnn.Sequential = classifier.sequential()
        self._soft_max: tnn.Softmax = tnn.Softmax(1)

    @override
    def forward(self, x: torch.Tensor):  # noqa: ANN201
        x = self._model_part(x)  # pyright: ignore[reportAny]
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        x = self._classifier(x)  # pyright: ignore[reportAny]
        x = self._soft_max(x)  # pyright: ignore[reportAny]
        return x
