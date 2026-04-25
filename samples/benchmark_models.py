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
        "-m", "--models-folder", type=Path, default=Path("./models"),
        help="Path to folder where models will be saved"
    )
    parser.add_argument(
        "--skip-configs", type=str, nargs="+", default=[],
        help="Skip configs with given names (without .toml extension)"
    )
    return parser


def get_config_files(configs_folder: Path, skip_names: set[str]) -> list[Path]:
    """Return sorted list of .toml config files, skipping those in skip_names."""
    if not configs_folder.exists():
        raise FileNotFoundError(f"Configs folder {configs_folder} does not exist.")
    all_configs = sorted(configs_folder.glob("*.toml"))
    filtered = [cfg for cfg in all_configs if cfg.stem not in skip_names]
    if not filtered:
        raise FileNotFoundError(
            f"No .toml files found in {configs_folder} after skipping."
        )
    return filtered


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
        "-m", str(args.models_folder),
    ]
    subprocess.run(cmd, check=True, capture_output=False)


def main() -> None:
    """Main entry point: run all experiments and collect results."""
    args = create_argparser().parse_args()
    skip_set = set(args.skip_configs)
    config_files = get_config_files(args.configs_folder, skip_set)

    if skip_set:
        logger.info(f"Skipping {len(skip_set)} config(s): {', '.join(skip_set)}")
    logger.info(f"Found {len(config_files)} config files:")
    for cfg in config_files:
        logger.info(f"  {cfg.name}")

    output_csv = args.output

    for cfg in config_files:
        model_name = cfg.stem
        logger.info(f"\n=== Running experiment for {model_name} ===")
        try:
            run_experiment(cfg, args, output_csv)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Experiment {cfg.name} failed: {e}")
            logger.info("Continuing with next config...")

    logger.info(f"\nAll experiments finished. Results saved to {output_csv}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()