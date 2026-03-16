import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
import torchvision.transforms.v2 as tt2
from torch.utils.data import DataLoader
from torchinfo import summary

from dataset import Dataset
from metrics import ModelMetrics
from training_config import TrainingConfig, load_config
from utils import TensorShape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(loader: DataLoader, device: torch.device, config: TrainingConfig) -> None:
    for epoch in range(config.epochs):
        config.network.train()
        logger.info(f"Epoch number {epoch} starts")
        for _, (images, labels) in enumerate(loader):
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            outputs = config.network(images)
            loss = config.loss_function(outputs, labels)

            config.optimizer.zero_grad()
            loss.backward()
            _ = config.optimizer.step()
        metrics = ModelMetrics.from_model(loader, config.network, device)
        logger.info(f"Epoch number {epoch} ends")
        logger.info(f"Epoch accuracy: {metrics.accuracy()}")
        logger.info(f"Epoch loss: {loss.item()}")


def main(
    train_dataset_path: Path, test_dataset_path: Path, config_path: Path
) -> None:  # TODO: think about how remove idiot suffix _path
    target_shape = (227, 227)
    input_transform = tt2.Compose([tt2.Resize(target_shape)])
    train_dataset = Dataset(train_dataset_path, input_transform)
    test_dataset = Dataset(test_dataset_path, input_transform)

    config = load_config(config_path, TensorShape(*target_shape, 3))

    train_loader = DataLoader(
        train_dataset,
        config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        config.batch_size,
        shuffle=False,
    )

    network_summary = summary(config.network, verbose=0, depth=5, col_names=[])

    logger.info(f"\n{format_torchsummary(str(network_summary))}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.network = config.network.to(device)

    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of batches: {len(train_loader)}")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate {config.learning_rate:.4f}")
    logger.info(f"Number of epochs: {config.epochs}")
    logger.info(f"Loss function: {config.loss_function.__class__.__name__}")
    logger.info(f"Optimizer: {config.optimizer.__class__.__name__}")

    logger.info("Start of training")
    train(train_loader, device, config)
    logger.info("End of training")

    logger.info("Start of testing")
    metrics = ModelMetrics.from_model(test_loader, config.network, device)
    logger.info(f"Testing accuracy: {metrics.accuracy():.4f}")
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
