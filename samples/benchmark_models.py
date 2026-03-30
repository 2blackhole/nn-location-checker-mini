#!/usr/bin/env python3
"""
Run experiments for all TOML configs in the configs/ folder.
Collect results into a single CSV and generate comparison plots.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

src_dir = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_dir))

from model_segment import SupportedModels

def create_argparser():
    parser = argparse.ArgumentParser(
        description="Run all experiments from TOML files in a directory"
    )
    parser.add_argument(
        "-trd", "--train_dataset", type=Path, required=True,
        help="Path to train dataset"
    )
    parser.add_argument(
        "-ted", "--test_dataset", type=Path, required=True,
        help="Path to test dataset"
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
        "-lf", "--log-folder", type=Path, default=Path("./logs/"),
        help="Path to log folder"
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=2, default=(500, 500),
        help="Image size (height width)"
    )
    return parser

def is_valid_model_name(name: str) -> bool:
    """Check if model name exists in SupportedModels."""
    try:
        p = SupportedModels[name.upper()]
        return True
    except KeyError:
        return False

def run_experiment(config_path: Path, args, output_csv: Path) -> None:
    """Run run_experiment.py for a single config and append to CSV."""
    subprocess.run(
        [
            sys.executable,
            Path(__file__).resolve().parent / "run_experiment.py",
            "-trd", str(args.train_dataset),
            "-ted", str(args.test_dataset),
            "-c", str(config_path),
            "-o", str(output_csv),
            "-lf", str(args.log_folder),
            "-s", str(args.size[0]), str(args.size[1]),
        ],
        check=True,
        capture_output=False,
    )
def read_csv_to_arrays(csv_path: Path):
    donors = []
    accuracy = []
    macro_f1 = []
    avg_time = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('accuracy') or not row.get('macro_f1') or not row.get('avg_time_per_image'):
                print(f"Warning: Skipping row with missing data: {row}")
                continue
            try:
                donors.append(row['donor'])
                accuracy.append(float(row['accuracy']))
                macro_f1.append(float(row['macro_f1']))
                avg_time.append(float(row['avg_time_per_image']))
            except ValueError as e:
                print(f"Warning: Skipping row due to conversion error: {row} - {e}")
                continue
    return donors, accuracy, macro_f1, avg_time

def plot_results(donors, accuracy, macro_f1, avg_time):
    """Generate bar charts for metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Comparison", fontsize=16)

    x = np.arange(len(donors))
    width = 0.6

    # Accuracy
    ax = axes[0, 0]
    ax.bar(x, accuracy, width)
    ax.set_xticks(x)
    ax.set_xticklabels(donors, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')

    # Macro F1
    ax = axes[0, 1]
    ax.bar(x, macro_f1, width)
    ax.set_xticks(x)
    ax.set_xticklabels(donors, rotation=45, ha='right')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Macro F1')

    # Average time per image
    ax = axes[1, 0]
    ax.bar(x, avg_time, width)
    ax.set_xticks(x)
    ax.set_xticklabels(donors, rotation=45, ha='right')
    ax.set_ylabel('Seconds')
    ax.set_title('Average Time per Image')

    # FPS (images per second)
    fps = 1 / np.array(avg_time)
    ax = axes[1, 1]
    ax.bar(x, fps, width)
    ax.set_xticks(x)
    ax.set_xticklabels(donors, rotation=45, ha='right')
    ax.set_ylabel('Images/sec')
    ax.set_title('Throughput (FPS)')

    plt.tight_layout()
    return fig

def save_metric_plot(donors, values, ylabel, title, filename):
    plt.figure(figsize=(8, 6))
    x = np.arange(len(donors))
    plt.bar(x, values)
    plt.xticks(x, donors, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    args = create_argparser().parse_args()

    configs_folder = args.configs_folder
    if not configs_folder.exists():
        print(f"Configs folder {configs_folder} does not exist.")
        return

    config_files = sorted(configs_folder.glob("*.toml"))
    if not config_files:
        print(f"No .toml files found in {configs_folder}")
        return

    print(f"Found {len(config_files)} config files:")
    for cfg in config_files:
        print(f"  {cfg.name}")

    output_csv = args.output

    for cfg in config_files:
        model_name = cfg.stem
        if not is_valid_model_name(model_name):
            print(f"Warning: model name '{model_name}' not found in SupportedModels, continuing anyway.")
        print(f"\n=== Running experiment for {model_name} ===")
        run_experiment(cfg, args, output_csv)

    print(f"\nAll experiments finished. Results saved to {output_csv}")

    if not output_csv.exists():
        print("No results CSV found, skipping plots.")
        return

    donors, accuracy, macro_f1, avg_time = read_csv_to_arrays(output_csv)

    fig = plot_results(donors, accuracy, macro_f1, avg_time)
    plot_path = Path("benchmark_plots.png")
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()