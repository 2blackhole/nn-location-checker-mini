import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch

from classification_network import ClassificationNetwork
from classifier import Classifier
from json_loader import ModuleLoader
from model_segment import ModelSegment, SupportedModels
from utils import TensorShape

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


def load_config(file: Path, input_shape: TensorShape) -> TrainingConfig:
    with file.open("rb") as configuration_file:
        config = tomllib.load(configuration_file)

    macro_p = config["macro_parameters"]
    model_p = config["model"]
    optimizer_p = config["optimizer"]
    loss_p = config["loss_function"]

    model = SupportedModels[model_p["name"].upper()]
    segment = ModelSegment(model, slice(model_p.get("start", 0), model_p["end"]))

    output_shape = segment.compute_shape(input_shape)
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
