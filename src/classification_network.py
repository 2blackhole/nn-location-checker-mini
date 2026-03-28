"""Full classification network.

This module wires a `model_segment.ModelSegment` together with a
`classifier.Classifier` into a single `torch.nn.Module`
ready for training or evaluation.

Classes:
    ClassificationNetwork: Model segment + classifier.

Functions:
    train_model: Run a training loop over the network.
    test_model: Evaluate the network on a dataset.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, override

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader

from classifier import Classifier
from metrics import ModelMetrics, Seconds
from model_segment import ModelSegment

if TYPE_CHECKING:
    from training_config import TrainingConfig

__all__ = ["ClassificationNetwork", "test_model", "train_model"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def test_model(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8], Seconds]:
    model.eval()
    total_time = 0.0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_start_time = time.time()
            outputs = model(images)
            batch_time = time.time() - batch_start_time
            total_time += batch_time
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(dim=1)
            all_predictions.append(predicted)
            all_labels.append(labels)
    total_predictions = torch.cat(all_predictions).numpy().astype(np.int8)
    total_labels = torch.cat(all_labels).numpy().astype(np.int8)
    return total_labels, total_predictions, total_time


def train_model(
    loader: DataLoader, device: torch.device, config: TrainingConfig
) -> None:
    for epoch in range(config.epochs):
        config.network.train()
        logger.info(f"Epoch number {epoch} starts")
        for _, (images, labels) in enumerate(loader):
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            outputs = config.network(images)
            loss = config.loss_function(outputs, labels)

            if torch.isnan(loss):
                message = "NaN loss detected, skipping batch"
                logger.critical(message)
                raise RuntimeError(message)

            config.optimizer.zero_grad()
            loss.backward()
            _ = config.optimizer.step()
        metrics = ModelMetrics(*test_model(loader, config.network, device))
        logger.info(f"Epoch number {epoch} ends")
        logger.info(f"Epoch accuracy: {metrics.accuracy()}")
        logger.info(f"Epoch loss: {loss.item()}")
