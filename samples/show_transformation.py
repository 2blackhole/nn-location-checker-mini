import argparse
from pathlib import Path
from random import choice
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
from matplotlib import pyplot as plot

from dataset import Dataset
from model_register import load_model_internals, lookup_model


def create_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    _ = argparser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset",
    )
    _ = argparser.add_argument(
        "-n",
        "--network",
        default="alexnet",
        help="Name of a network to get transformation",
    )
    return argparser


def show_images(images: tuple[torch.Tensor, torch.Tensor]) -> None:
    origin_image, transformed_image = images
    transformed_image = transformed_image.clamp(0.0, 1.0)
    figsize = (10, 10)
    fig, axes = plot.subplots(1, 2, figsize=figsize)
    _ = fig.suptitle("Transformation of images")
    _ = plot.setp(plot.gcf().get_axes(), xticks=[], yticks=[])
    axes[0].imshow(origin_image.permute(1, 2, 0).numpy())
    axes[1].imshow(transformed_image.permute(1, 2, 0).numpy())
    fig.tight_layout()
    plot.show()


def main(dataset_path: Path, network: str) -> None:
    dataset = Dataset(dataset_path)
    image, _ = choice(dataset)
    transformation = load_model_internals(lookup_model(network)).transform
    resize = transformation.transforms[0]
    show_images((resize(image), transformation(image)))


if __name__ == "__main__":
    argparser = create_argparser()
    arguments = argparser.parse_args()
    main(arguments.dataset, arguments.network)
