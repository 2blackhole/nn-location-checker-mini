"""Full classification network.

This module provides `ClassificationNetwork`, which wires a
`model_segment.ModelSegment` together with a
`classifier.Classifier` into a single `torch.nn.Module`
ready for training or eval.

"""

from typing import override

import torch
import torch.nn as tnn

from classifier import Classifier
from model_segment import ModelSegment

__all__ = ["ClassificationNetwork"]


class ClassificationNetwork(tnn.Module):
    """A classification network combining a existing model segment and a classifier.

    Passes input through a `model_segment.ModelSegment`,
    flattens the output if needed, then runs it through a
    `classifier.Classifier` head.

    """

    def __init__(self, model_part: ModelSegment, classifier: Classifier) -> None:
        """Initialize the network.

        Args:
            model_part: A slice of a exsisting model
                from `model_segment.SupportedModels`.
            classifier: Classifier, rewired to connect to the
                output dimension of ``model_part``.
        """
        super().__init__()

        self._model_part: ModelSegment = model_part
        self._classifier: tnn.Sequential = classifier.sequential()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._model_part(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        return self._classifier(x)
