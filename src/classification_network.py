from typing import override

import torch
import torch.nn as tnn

from classifier import Classifier
from model_segment import ModelSegment


class ClassificationNetwork(tnn.Module):
    def __init__(self, model_part: ModelSegment, classifier: Classifier) -> None:
        super().__init__()

        self._model_part: ModelSegment = model_part
        self._classifier: tnn.Sequential = classifier.sequential()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._model_part(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        return self._classifier(x)
