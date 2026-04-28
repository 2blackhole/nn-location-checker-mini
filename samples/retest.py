import argparse
import logging
import sys
from pathlib import Path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys.path.append(str(src_directory))

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from classification_network import test_model
from dataset import Dataset, Marker
from logger import configure_logger
from metrics import QualityMetrics, TimeMetrics
from training_config import load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def configure_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    _ = argparser.add_argument(
        "-d", "--dataset", type=Path, required=True, help="Dataset to test network"
    )
    _ = argparser.add_argument(
        "-c", "--config", type=Path, required=True, help="Path to network configuration"
    )
    _ = argparser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to networks's weights"
    )
    _ = argparser.add_argument(
        "-l", "--log-folder", type=Path, default="./logs", help="Path to log folder"
    )
    _ = argparser.add_argument(
        "-n",
        "--log-name",
        default="retest.log",
        help="Name of the log file with extension",
    )
    return argparser


def format_torchsummary(summary: str) -> str:
    lines = summary.splitlines()
    end = next(i for i, line in enumerate(lines) if i > 3 and line.endswith("="))
    return "\n".join(lines[3:end])


def main(dataset: Path, config: Path, weights: Path) -> None:
    cfg = load_config(config)
    data_loader = DataLoader(
        Dataset(dataset, cfg.transform), cfg.batch_size, shuffle=False
    )

    network_summary = summary(cfg.network, verbose=0, depth=5, col_names=[])
    logger.info(f"\n{format_torchsummary(str(network_summary))}")
    logger.info(f"Donor: {cfg.donor}")
    logger.info(f"Segment: {cfg.segment_start}:{cfg.segment_end}")
    logger.info(f"Classifier: {cfg.classifier}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.network.load_state_dict(torch.load(weights, weights_only=True))
    cfg.network = cfg.network.to(device)

    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Number of batches: {len(data_loader)}")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate {cfg.learning_rate}")
    logger.info(f"Number of epochs: {cfg.epochs}")
    logger.info(f"Loss function: {cfg.loss_function.__class__.__name__}")
    logger.info(f"Optimizer: {cfg.optimizer.__class__.__name__}")
    logger.info("Start of testings")
    total_labels, total_predictions, total_time = test_model(
        data_loader, cfg.network, device
    )
    quality_metrics = QualityMetrics(total_labels, total_predictions)
    time_metrics = TimeMetrics(total_labels.size, total_time)
    logger.info(f"Accuracy: {quality_metrics.accuracy():.4f}")
    f1_scores = [round(quality_metrics.f1_score(label), 4) for label in Marker]
    logger.info(f"Macro f1 per class: {f1_scores}")
    logger.info(f"Macro f1: {quality_metrics.f1_score():.4f}")
    logger.info(f"Average time per image: {time_metrics.avg_time_per_image():.4f} s")
    logger.info(f"Classification speed: {time_metrics.fps():.4f} images/s")
    logger.info("End of testing")


if __name__ == "__main__":
    argparser = configure_argparser()
    arguments = argparser.parse_args()
    configure_logger(arguments.log_folder, arguments.log_name)
    main(arguments.dataset, arguments.config, arguments.model)
