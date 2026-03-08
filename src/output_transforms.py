import time

import torch
from torch.utils.data import DataLoader

from build_cnn import CNnetwork


def get_accuracy(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: CNnetwork,
    device: torch.device,
) -> tuple[float, float]:
    tp = 0
    n = 0
    total_time = 0.0
    with torch.no_grad():
        for images, labels in data_loader:  # pyright: ignore[reportAny]
            images = images.requires_grad_().to(device)  # pyright: ignore[reportAny]
            labels = labels.to(device)  # pyright: ignore[reportAny]
            batch_start_time = time.time()
            outputs = model(images)  # pyright: ignore[reportAny]
            batch_time = time.time() - batch_start_time
            _, predicted = torch.max(outputs.data, 1)  # pyright: ignore[reportAny]
            n += labels.size(0)  # pyright: ignore[reportAny]
            tp += (predicted == labels).sum()  # pyright: ignore[reportAny]
            total_time += batch_time
    avg_time_per_image = float(total_time / n)

    accuracy = float(tp / n)
    if not isinstance(accuracy, float):
        raise RuntimeError(f"Accuracy is not a float: {type(accuracy)}")

    if not isinstance(avg_time_per_image, float):
        raise RuntimeError(f"Avg time is not float: {type(avg_time_per_image)}")

    return accuracy, avg_time_per_image
