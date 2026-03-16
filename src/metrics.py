import time
from typing import Self

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from classification_network import ClassificationNetwork
from dataset import Marker

type Seconds = float


def _run_model(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> tuple[npt.NDArray[np.int32], Seconds]:
    model.eval()
    confusion_matrix = np.zeros((len(Marker), len(Marker)), dtype=np.int32)
    total_time = 0.0
    with torch.no_grad():
        for images, labels in data_loader:  # pyright: ignore[reportAny]
            images = images.requires_grad_().to(device)  # pyright: ignore[reportAny]
            labels = labels.to(device)  # pyright: ignore[reportAny]
            batch_start_time = time.time()
            outputs = model(images)  # pyright: ignore[reportAny]
            batch_time = time.time() - batch_start_time
            total_time += batch_time
            _, predicted = torch.max(outputs.data, 1)  # pyright: ignore[reportAny]
            for predict, label in zip(predicted, labels, strict=True):
                confusion_matrix[label][predict] += 1

    return confusion_matrix, total_time


class ModelMetrics:
    def __init__(
        self, confusion_matrix: npt.NDArray[np.int32], total_time: Seconds
    ) -> None:
        num_of_classes = len(Marker)
        if confusion_matrix.shape != (num_of_classes, num_of_classes):
            raise ValueError(
                f"Confusion matrix must be a square matrix {len(Marker)}x{len(Marker)}"
            )

        self._confusion_matrix: npt.NDArray[np.int32] = np.zeros(
            (num_of_classes, num_of_classes), dtype=np.int32
        )

        for i in range(num_of_classes):
            self._confusion_matrix[i] = confusion_matrix[i]

        self._total_time: Seconds = total_time

    @classmethod
    def from_model(
        cls,
        data_loader: DataLoader[tuple[torch.Tensor, int]],
        model: ClassificationNetwork,
        device: torch.device,
    ) -> Self:
        confusion_matrix, total_time = _run_model(data_loader, model, device)

        return cls(confusion_matrix, total_time)

    def accuracy(self) -> float:
        return np.diag(self._confusion_matrix).sum() / self._confusion_matrix.sum()

    def precision(self, label: Marker | None = None) -> float:
        if label is None:
            precisions = [self.precision(label) for label in Marker]
            return float(np.mean(precisions))
        label_i = label.value
        true_positives = self._confusion_matrix[label_i, label_i]
        false_positives = self._confusion_matrix[..., label_i].sum() - true_positives
        return (
            0.0
            if true_positives + false_positives == 0
            else true_positives / (true_positives + false_positives)
        )

    def recall(self, label: Marker | None = None) -> float:
        if label is None:
            recalls = [self.recall(label) for label in Marker]
            return float(np.mean(recalls))

        label_i = label.value
        true_positives = self._confusion_matrix[label_i, label_i]
        false_negatives = self._confusion_matrix[label_i, ...].sum() - true_positives

        return (
            0.0
            if true_positives + false_negatives == 0
            else true_positives / (true_positives + false_negatives)
        )

    def avg_time_per_image(self) -> Seconds:
        return self._total_time / self._confusion_matrix.sum()

    def f1_score(self, label: Marker | None = None) -> float:
        if label is None:
            f1_scores = [self.f1_score(label) for label in Marker]
            return float(np.mean(f1_scores))
        label_i = label.value
        true_positives = self._confusion_matrix[label_i, label_i]
        false_positives = self._confusion_matrix[..., label_i].sum() - true_positives
        false_negatives = self._confusion_matrix[label_i, ...].sum() - true_positives

        return (2 * true_positives) / (
            2 * true_positives + false_positives + false_negatives
        )

    @property
    def confusion_matrix(self) -> npt.NDArray[np.int32]:
        return self._confusion_matrix

    @property
    def total_time(self) -> Seconds:
        return self._total_time
