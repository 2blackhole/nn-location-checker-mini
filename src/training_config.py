"""Training configuration loader for the training pipeline.

This module provides `TrainingConfig`, a dataclass holding all components
needed for a training run, and `load_config` for constructing it from a
TOML file.

Intended usage::

    config = load_config(Path("config.toml"), TensorShape(227, 227, 3))

    for epoch in range(config.epochs):
        train(config.network, config.optimizer, config.loss_function)
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torchvision.transforms.v2 as tt2

from classification_network import ClassificationNetwork
from classifier import Classifier
from json_loader import ModuleLoader
from model_register import load_model_internals, lookup_model
from model_segment import ModelSegment
from tensor_shape import TensorShape

__all__ = ["TrainingConfig", "load_config"]


@dataclass
class TrainingConfig:
    donor: str
    transform: tt2.Transform | None
    classifier: Classifier
    batch_size: int
    epochs: int
    network: ClassificationNetwork
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    learning_rate: float
    segment_start: int
    segment_end: int
    target_shape: TensorShape


def load_config(file: Path) -> TrainingConfig:
    with file.open("rb") as configuration_file:
        config = tomllib.load(configuration_file)

    macro_p = config["macro_parameters"]
    model_p = config["model"]
    optimizer_p = config["optimizer"]
    loss_p = config["loss_function"]

    model = lookup_model(model_p["name"])
    internals = load_model_internals(model)
    height, width = cast(tt2.CenterCrop, internals.transform.transforms[1]).size
    target_shape = TensorShape(height, width, 3)
    segment_start = model_p.get("start", 0)
    segment_end = model_p["end"]
    segment = ModelSegment(internals.modules, slice(segment_start, segment_end))
    output_shape = segment.compute_shape(target_shape)
    classifier = ModuleLoader(model_p["classifier"]).load(output_shape)
    network = ClassificationNetwork(segment, classifier)

    optimizer = getattr(torch.optim, optimizer_p["name"])(
        network.parameters(), lr=optimizer_p["learning_rate"]
    )
    loss_fn = getattr(torch.nn, loss_p["name"])()

    return TrainingConfig(
        donor=model_p["name"],
        transform=internals.transform,
        classifier=classifier,
        batch_size=macro_p["batch_size"],
        epochs=macro_p["epochs"],
        network=network,
        optimizer=optimizer,
        loss_function=loss_fn,
        learning_rate=optimizer_p["learning_rate"],
        segment_start=segment_start,
        segment_end=segment_end,
        target_shape=target_shape,
    )
