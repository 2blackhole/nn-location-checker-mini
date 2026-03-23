"""Classification metrics aggregation over full evaluation runs.

This module provides `ModelMetrics`, a dataclass that computes accuracy,
precision, recall, F1, and confusion matrix from collected labels and
predictions, with support for both macro-averaged and per-class scores.

Intended usage::

    metrics = ModelMetrics(
        labels=labels,
        predictions=predictions,
        total_time=elapsed,
    )
    print(metrics.accuracy())
    print(metrics.f1_score(Marker.KREMLIN))
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import Marker

__all__ = ["ModelMetrics", "Seconds"]

type Seconds = float


@dataclass
class ModelMetrics:
    labels: npt.NDArray[np.int8]
    predictions: npt.NDArray[np.int8]
    total_time: Seconds
    ALL_LABELS: ClassVar[list[int]] = [label.value for label in Marker]

    def accuracy(self) -> float:
        return accuracy_score(self.labels, self.predictions)

    def precision(self, label: Marker | None = None) -> float:
        result = 0.0
        if label is None:
            result = precision_score(
                self.labels, self.predictions, average="macro", labels=self.ALL_LABELS
            )
        else:
            result = precision_score(
                self.labels, self.predictions, average=None, labels=self.ALL_LABELS
            )[label.value]

        return float(result)

    def recall(self, label: Marker | None = None) -> float:
        result = 0.0
        if label is None:
            result = recall_score(
                self.labels, self.predictions, average="macro", labels=self.ALL_LABELS
            )
        else:
            result = recall_score(
                self.labels, self.predictions, average=None, labels=self.ALL_LABELS
            )[label.value]

        return float(result)

    def avg_time_per_image(self) -> float:
        return float(self.total_time / self.predictions.size)

    def f1_score(self, label: Marker | None = None) -> float:
        result = 0.0
        if label is None:
            result = f1_score(
                self.labels, self.predictions, average="macro", labels=self.ALL_LABELS
            )
        else:
            result = f1_score(
                self.labels, self.predictions, average=None, labels=self.ALL_LABELS
            )[label.value]

        return float(result)

    def confusion_matrix(self) -> npt.NDArray[np.int32]:
        return confusion_matrix(self.labels, self.predictions, labels=self.ALL_LABELS)
