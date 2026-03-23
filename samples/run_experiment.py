import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys.path.append(str(src_directory))

from experiment import Experiment, ExperimentCSVHandler


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

    return argparser


def venv_exists() -> bool:
    project_root = Path(__file__).resolve().parents[1]

    python = project_root.joinpath(".venv/bin/python")

    return python.exists()


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
        stderr=asyncio.subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as training:
        for line in training.stderr:  # ty:ignore[not-iterable]
            print(line.strip())
            experiment.update(line)

    return experiment


def main(arguments: argparse.Namespace) -> None:
    train_dataset = arguments.train_dataset
    test_dataset = arguments.test_dataset or arguments.train_dataset
    config = arguments.config

    experiment = run(train_dataset, test_dataset, config)
    with ExperimentCSVHandler(arguments.output) as output:
        output.writerow(experiment)


if __name__ == "__main__":
    parser = create_argparser()
    arguments = parser.parse_args()
    main(arguments)
