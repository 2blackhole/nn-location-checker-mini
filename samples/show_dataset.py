import argparse
from pathlib import Path
from random import sample
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
import torchvision.transforms.v2 as tt2
from matplotlib import pyplot as plot

from dataset import Dataset, Marker


def show_images(dataset_picks: list[tuple[torch.Tensor, int]]) -> None:
    num_showed_imgs_x = 5
    num_showed_imgs_y = 5

    figsize = (10, 10)
    fig, axes = plot.subplots(num_showed_imgs_y, num_showed_imgs_x, figsize=figsize)
    _ = fig.suptitle("Dataset images")
    _ = plot.setp(plot.gcf().get_axes(), xticks=[], yticks=[])
    for i, ax in enumerate(axes.flat):
        if i < len(dataset_picks):
            img = dataset_picks[i][0].byte().permute(1, 2, 0).numpy()
            label_idx = dataset_picks[i][1]
            label_name = Marker(label_idx).name.capitalize().replace("_", " ")
            ax.imshow(img)
            ax.set_xlabel(label_name, fontsize=8)

    fig.tight_layout()
    plot.show()


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
        "-s",
        "--size",
        type=int,
        nargs=2,
        default=(500, 500),
        help="Size of images",
    )

    return argparser


def main(dataset_path: Path, image_size: tuple[int, int]) -> None:
    dataset = Dataset(dataset_path, tt2.Resize(image_size))
    random_25_idx = sample(range(0, len(dataset)), 25)
    dataset_picks = [dataset[idx] for idx in random_25_idx]
    show_images(dataset_picks)


if __name__ == "__main__":
    argparser = create_argparser()
    arguments = argparser.parse_args()

    dataset: Path = arguments.dataset
    image_size: tuple[int, int] = tuple(arguments.size)
    main(dataset, image_size)
