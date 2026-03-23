import argparse
import logging
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))


import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms.v2 as tt2
from torch.utils.data import DataLoader
from torchinfo import summary

from classification_network import ClassificationNetwork
from dataset import Dataset, Marker
from metrics import ModelMetrics, Seconds
from training_config import TrainingConfig, load_config
from utils import TensorShape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_model(
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


def train(loader: DataLoader, device: torch.device, config: TrainingConfig) -> None:
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
        metrics = ModelMetrics(*run_model(loader, config.network, device))
        logger.info(f"Epoch number {epoch} ends")
        logger.info(f"Epoch accuracy: {metrics.accuracy()}")
        logger.info(f"Epoch loss: {loss.item()}")


def main(train_dataset: Path, test_dataset: Path, config: Path) -> None:
    target_shape = (227, 227)
    input_transform = tt2.Compose([tt2.Resize(target_shape)])
    train_data = Dataset(train_dataset, input_transform)
    test_data = Dataset(test_dataset, input_transform)

    cfg = load_config(config, TensorShape(*target_shape, 3))

    train_loader = DataLoader(
        train_data,
        cfg.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        cfg.batch_size,
        shuffle=False,
    )

    network_summary = summary(cfg.network, verbose=0, depth=5, col_names=[])

    logger.info(f"\n{format_torchsummary(str(network_summary))}")
    logger.info(f"Donor: {cfg.donor}")
    logger.info(f"Segment: {cfg.segment_start}:{cfg.segment_end}")
    logger.info(f"Classifier: {cfg.classifier}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.network = cfg.network.to(device)

    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Number of batches: {len(train_loader)}")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate {cfg.learning_rate}")
    logger.info(f"Number of epochs: {cfg.epochs}")
    logger.info(f"Loss function: {cfg.loss_function.__class__.__name__}")
    logger.info(f"Optimizer: {cfg.optimizer.__class__.__name__}")

    logger.info("Start of training")
    train(train_loader, device, cfg)
    logger.info("End of training")

    logger.info("Start of testing")
    metrics = ModelMetrics(*run_model(test_loader, cfg.network, device))
    logger.info(f"Accuracy: {metrics.accuracy():.4f}")
    logger.info(
        f"Macro f1 per class: {[round(metrics.f1_score(label), 4) for label in Marker]}"
    )
    logger.info(f"Macro f1: {metrics.f1_score():.4f}")
    logger.info(f"Average time per image: {metrics.avg_time_per_image():.4f} s")
    logger.info(
        f"Classification speed: {1 / metrics.avg_time_per_image():.4f} images/s"
    )
    logger.info("End of testing")


def format_torchsummary(summary: str) -> str:
    lines = summary.splitlines()
    end = next(i for i, line in enumerate(lines) if i > 3 and line.endswith("="))
    return "\n".join(lines[3:end])


def configure_logger() -> None:
    to_logs_folder = Path("./logs/")
    if not to_logs_folder.exists():
        raise RuntimeError("Please, init logs folder in root of the project")
    file_handler = TimedRotatingFileHandler(
        to_logs_folder.joinpath("latest.log"), "midnight", encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    format_template = "%(asctime)s %(levelname)s:%(message)s"
    date_template = "%d/%m/%Y %H:%M:%S"
    logging.basicConfig(
        format=format_template,
        datefmt=date_template,
        handlers=[console_handler, file_handler],
    )


def create_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    _ = argparser.add_argument(
        "-trd",
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to train dataset",
    )
    _ = argparser.add_argument(
        "-ted", "--test_dataset", type=Path, help="Path to test dataset", default=None
    )
    _ = argparser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to train_config.toml",
    )

    return argparser


if __name__ == "__main__":
    configure_logger()
    parser = create_argparser()
    arguments = parser.parse_args()
    train_dataset = arguments.train_dataset
    test_dataset = arguments.test_dataset or arguments.train_dataset
    config = arguments.config
    main(train_dataset, test_dataset, config)
