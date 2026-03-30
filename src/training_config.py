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

import torch

from classification_network import ClassificationNetwork
from classifier import Classifier
from json_loader import ModuleLoader
from model_segment import ModelSegment, SupportedModels
from tensor_shape import TensorShape

__all__ = ["TrainingConfig", "load_config"]


@dataclass
class TrainingConfig:
    donor: str
    classifier: Classifier
    batch_size: int
    epochs: int
    network: ClassificationNetwork
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    learning_rate: float
    segment_start: int
    segment_end: int

def get_output_shape(segment: ModelSegment, input_shape: TensorShape) -> int:
    """Calculate actual output size by passing a dummy tensor through the segment."""
    with torch.no_grad():
        dummy = torch.zeros(1, input_shape.channels, input_shape.height, input_shape.width)
        out = segment(dummy)
        if len(out.shape) > 2:
            out = torch.flatten(out, 1)
        return out.shape[1]

def load_config(file: Path, input_shape: TensorShape) -> TrainingConfig:
    with file.open("rb") as configuration_file:
        config = tomllib.load(configuration_file)

    macro_p = config["macro_parameters"]
    model_p = config["model"]
    optimizer_p = config["optimizer"]
    loss_p = config["loss_function"]

    model = SupportedModels[model_p["name"].upper()]
    segment = ModelSegment(model, slice(model_p.get("start", 0), model_p["end"]))

    """
    Idk this was not working for me 
    INFO:RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x50176 and 1024x1000)
    """
    # output_shape = segment.compute_shape(input_shape)


    """
    So I searched for this one func and it's kinda works
    """
    output_shape = get_output_shape(segment, input_shape)



    classifier = ModuleLoader(model_p["classifier"]).load(output_shape)
    network = ClassificationNetwork(segment, classifier)

    optimizer = getattr(torch.optim, optimizer_p["name"])(
        network.parameters(), lr=optimizer_p["learning_rate"]
    )
    loss_fn = getattr(torch.nn, loss_p["name"])()

    return TrainingConfig(
        donor=model_p["name"],
        classifier=classifier,
        batch_size=macro_p["batch_size"],
        epochs=macro_p["epochs"],
        network=network,
        optimizer=optimizer,
        loss_function=loss_fn,
        learning_rate=optimizer_p["learning_rate"],
        segment_start=model_p.get("start", 0),
        segment_end=model_p["end"],
    )
