"""
Run experiments for all TOML configs in the configs/ folder.
Collect results into a single CSV.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_dir))

from model_segment import SupportedModels

logger = logging.getLogger(__name__)


def create_argparser() -> argparse.ArgumentParser:
    """Create argument parser for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run all experiments from TOML files in a directory"
    )
    parser.add_argument(
        "-trd", "--train_dataset", type=Path, required=True,
        help="Path to the train dataset"
    )
    parser.add_argument(
        "-ted", "--test_dataset", type=Path, required=True,
        help="Path to the test dataset"
    )
    parser.add_argument(
        "-cf", "--configs_folder", type=Path, default=Path("../configs"),
        help="Folder containing TOML config files (default: ../configs)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("benchmark_results.csv"),
        help="Output CSV file (default: benchmark_results.csv)"
    )
    parser.add_argument(
        "-lf", "--log-folder", type=Path, default=Path("./logs"),
        help="Path to the log folder"
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=2, default=(500, 500),
        help="Image size (height width)"
    )
    return parser


def is_valid_model_name(name: str) -> bool:
    """Check if model name exists in SupportedModels."""
    try:
        _ = SupportedModels[name.upper()]
        return True
    except KeyError:
        return False


def get_config_files(configs_folder: Path) -> list[Path]:
    """Return sorted list of .toml config files."""
    if not configs_folder.exists():
        raise FileNotFoundError(
            f"Configs folder {configs_folder} does not exist."
        )
    config_files = sorted(configs_folder.glob("*.toml"))
    if not config_files:
        raise FileNotFoundError(
            f"No .toml files found in {configs_folder}"
        )
    return config_files


def run_experiment(
    config_path: Path,
    args: argparse.Namespace,
    output_csv: Path,
) -> None:
    """Run run_experiment.py for a single config and append to CSV."""
    script_path = Path(__file__).resolve().parent / "run_experiment.py"
    cmd = [
        sys.executable,
        str(script_path),
        "-trd", str(args.train_dataset),
        "-ted", str(args.test_dataset),
        "-c", str(config_path),
        "-o", str(output_csv),
        "-lf", str(args.log_folder),
        "-s", str(args.size[0]), str(args.size[1]),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        logger.error(
            "Experiment failed for %s with exit code %d",
            config_path.name, e.returncode
        )
        raise RuntimeError(
            f"Experiment {config_path.name} failed, check logs"
        ) from e


def main() -> None:
    """Main entry point: run all experiments and collect results."""
    args = create_argparser().parse_args()
    config_files = get_config_files(args.configs_folder)

    logger.info("Found %d config files:", len(config_files))
    for cfg in config_files:
        logger.info("  %s", cfg.name)

    output_csv = args.output

    for cfg in config_files:
        model_name = cfg.stem
        if not is_valid_model_name(model_name):
            logger.warning(
                "Model name '%s' not found in SupportedModels, continuing anyway.",
                model_name
            )
        logger.info("\n=== Running experiment for %s ===", model_name)
        run_experiment(cfg, args, output_csv)

    logger.info("\nAll experiments finished. Results saved to %s", output_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()