import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys.path.append(str(src_directory))

from experiment import Experiment, ExperimentCSVHandler
from logger import configure_logger

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
        "-o",
        "--output",
        type=Path,
        default=Path("experiment_results.csv"),
        help="Path to output csv file",
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
        default="experiment.log",
        help="Name of the log file with extension",
    )

    return argparser


def venv_exists() -> bool:
    project_root = Path(__file__).resolve().parents[1]
    python = project_root.joinpath(".venv/bin/python")
    return python.exists()


_LOG_PREFIX = re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} \w+:(.*)", re.DOTALL)


def dedup_logger_output(message: str) -> str:
    m = _LOG_PREFIX.match(message)
    return m.group(1) if m else message


def run(train_dataset: Path, test_dataset: Path, config: Path) -> Experiment:
    if not venv_exists():
        raise RuntimeError("Create venv")

    experiment = Experiment()
    training_script = Path(__file__).resolve().parents[0].joinpath("train_model.py")
    with subprocess.Popen(
        [
            sys.executable,
            training_script,
            "-trd",
            str(train_dataset),
            "-ted",
            str(test_dataset),
            "-c",
            str(config),
        ],
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as training:
        for line in training.stderr:  # ty:ignore[not-iterable]
            logger.info(dedup_logger_output(line.rstrip()))
            experiment.update(line)

    return experiment


def main(arguments: argparse.Namespace) -> None:
    train_dataset = arguments.train_dataset
    test_dataset = arguments.test_dataset or arguments.train_dataset
    config = arguments.config

    experiment = run(train_dataset, test_dataset, config)
    try:
        with ExperimentCSVHandler(arguments.output) as output:
            output.writerow(experiment)
    except Exception as e:
        logger.critical(f"Can't write experiment to {arguments.output}")
        logger.exception(e)
        logger.info("Print experiment as dict in log-stream")
        logger.info(dict(experiment))


if __name__ == "__main__":
    parser = create_argparser()
    arguments = parser.parse_args()
    configure_logger(arguments.log_folder, arguments.log_name)
    main(arguments)
