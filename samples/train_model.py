import argparse
import logging
from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
import torchvision.transforms.v2 as tt2
from torch.utils.data import DataLoader
from torchinfo import summary

from classification_network import test_model, train_model
from dataset import Dataset, Marker
from logger import configure_logger
from metrics import QualityMetrics, TimeMetrics
from tensor_shape import TensorShape
from training_config import load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    _ = argparser.add_argument(
        "-lf",
        "--log-folder",
        type=Path,
        default=Path("./logs/"),
        help="Path to log folder",
    )
    _ = argparser.add_argument(
        "-ln",
        "--log-name",
        default="train.log",
        help="Name of the log file with extension",
    )
    _ = argparser.add_argument(
        "-s",
        "--size",
        type=int,
        nargs=2,
        default=(500, 500),
        help="Size of images",
    )

    return argparser


def setup_dataloaders(
    paths: tuple[Path, Path], batch_size: int, transform: tt2.Transform
) -> tuple[DataLoader, DataLoader]:
    train_data = Dataset(paths[0], transform)
    test_data = Dataset(paths[1], transform)
    return DataLoader(train_data, batch_size, shuffle=True), DataLoader(
        test_data, batch_size, shuffle=False
    )


def format_torchsummary(summary: str) -> str:
    lines = summary.splitlines()
    end = next(i for i, line in enumerate(lines) if i > 3 and line.endswith("="))
    return "\n".join(lines[3:end])


def main(
    train_dataset: Path, test_dataset: Path, config: Path, target_shape: tuple[int, int]
) -> None:
    input_transform = tt2.Compose([tt2.Resize(target_shape)])
    cfg = load_config(config, TensorShape(*target_shape, 3))
    train_loader, test_loader = setup_dataloaders(
        (train_dataset, test_dataset), cfg.batch_size, input_transform
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
    train_model(train_loader, device, cfg)
    logger.info("End of training")

    logger.info("Start of testing")
    total_labels, total_predictions, total_time = test_model(
        test_loader, cfg.network, device
    )
    quality_metrics = QualityMetrics(total_labels, total_predictions)
    time_metrics = TimeMetrics(total_labels.size, total_time)
    logger.info(f"Accuracy: {quality_metrics.accuracy():.4f}")
    f1_scores = [round(quality_metrics.f1_score(label), 4) for label in Marker]
    logger.info(f"Macro f1 per class: {f1_scores}")
    logger.info(f"Macro f1: {quality_metrics.f1_score():.4f}")
    logger.info(f"Average time per image: {time_metrics.avg_time_per_image():.4f} s")
    logger.info(
        f"Classification speed: {1 / time_metrics.avg_time_per_image():.4f} images/s"
    )
    logger.info("End of testing")


if __name__ == "__main__":
    parser = create_argparser()
    arguments = parser.parse_args()
    configure_logger(arguments.log_folder, arguments.log_name)
    train_dataset = arguments.train_dataset
    test_dataset = arguments.test_dataset or arguments.train_dataset
    config = arguments.config
    target_shape = arguments.size
    main(train_dataset, test_dataset, config, target_shape)
