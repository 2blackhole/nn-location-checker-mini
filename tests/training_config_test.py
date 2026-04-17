import json
from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))


import pytest
import torch
import torch.nn as tnn

from classification_network import ClassificationNetwork
from tensor_shape import TensorShape
from training_config import TrainingConfig, load_config


def write_toml(tmp_path: Path, content: str, name: str = "config.toml") -> Path:
    path = tmp_path.joinpath(name)
    path.write_text(content, encoding="utf-8")
    return path


def write_classifier_json(tmp_path: Path) -> Path:
    path = tmp_path.joinpath("classifier.json")
    path.write_text(
        json.dumps([{"type": "linear", "out": 15}]),
        encoding="utf-8",
    )
    return path


VALID_TOML_TEMPLATE = """
[macro_parameters]
batch_size = 32
epochs = 10

[model]
name = "alexnet"
end = 2
classifier = "{classifier_path}"
image_size = [200, 200]

[optimizer]
name = "SGD"
learning_rate = 0.001

[loss_function]
name = "CrossEntropyLoss"
"""


@pytest.fixture
def classifier_json(tmp_path: Path):
    return write_classifier_json(tmp_path)


@pytest.fixture
def valid_config(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json)
    return write_toml(tmp_path, content)


@pytest.fixture
def input_shape():
    return TensorShape(227, 227, 3)


def test_load_config_returns_training_config(valid_config: Path):
    config = load_config(valid_config)
    assert isinstance(config, TrainingConfig)


def test_load_config_batch_size(valid_config: Path):
    config = load_config(valid_config)
    assert config.batch_size == 32


def test_load_config_epochs(valid_config: Path):
    config = load_config(valid_config)
    assert config.epochs == 10


def test_load_config_learning_rate(valid_config: Path):
    config = load_config(valid_config)
    assert config.learning_rate == pytest.approx(0.001)


def test_load_config_segment_start_defaults_to_zero(valid_config: Path):
    config = load_config(valid_config)
    assert config.segment_start == 0


def test_load_config_segment_end(valid_config: Path):
    config = load_config(valid_config)
    assert config.segment_end == 2


def test_load_config_optimizer_type(valid_config: Path):
    config = load_config(valid_config)
    assert isinstance(config.optimizer, torch.optim.SGD)


def test_load_config_loss_function_type(valid_config: Path):
    config = load_config(valid_config)
    assert isinstance(config.loss_function, tnn.CrossEntropyLoss)


def test_load_config_network_is_classificaton_network(valid_config: Path):
    config = load_config(valid_config)
    assert isinstance(config.network, ClassificationNetwork)


def test_load_config_explicit_start(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json)
    content += "\nstart = 1\n"
    content = content.replace("[optimizer]", "start = 1\n\n[optimizer]").replace(
        "end = 2\nstart = 1", "start = 1\nend = 2"
    )
    path = write_toml(tmp_path, content)
    config = load_config(path)
    assert config.segment_start == 1


@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam", "AdamW"])
def test_load_config_optimizer_variants(
    tmp_path: Path, classifier_json: Path, optimizer_name: str
):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        'name = "SGD"', f'name = "{optimizer_name}"'
    )
    config = load_config(write_toml(tmp_path, content))
    assert isinstance(config.optimizer, getattr(torch.optim, optimizer_name))


@pytest.mark.parametrize("loss_name", ["CrossEntropyLoss", "NLLLoss"])
def test_load_config_loss_variants(
    tmp_path: Path, classifier_json: Path, loss_name: str
):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        'name = "CrossEntropyLoss"', f'name = "{loss_name}"'
    )
    config = load_config(write_toml(tmp_path, content))
    assert isinstance(config.loss_function, getattr(tnn, loss_name))


def test_missing_batch_size_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        "batch_size = 32\n", ""
    )
    with pytest.raises(KeyError):
        load_config(write_toml(tmp_path, content))


def test_missing_epochs_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        "epochs = 10\n", ""
    )
    with pytest.raises(KeyError):
        load_config(write_toml(tmp_path, content))


def test_missing_model_end_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        "end = 2\n", ""
    )
    with pytest.raises(KeyError):
        load_config(write_toml(tmp_path, content))


def test_missing_learning_rate_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        "learning_rate = 0.001\n", ""
    )
    with pytest.raises(KeyError):
        load_config(write_toml(tmp_path, content))


def test_missing_classifier_path_raises(tmp_path: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path="nonexistent.json")
    with pytest.raises(ValueError, match="does not exist"):
        load_config(write_toml(tmp_path, content))


def test_invalid_model_name_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        'name = "alexnet"', 'name = "nonexistent_model"'
    )
    with pytest.raises(ValueError, match="Unknown model"):
        load_config(write_toml(tmp_path, content))


def test_invalid_optimizer_name_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        'name = "SGD"', 'name = "InvalidOptimizer"'
    )
    with pytest.raises(AttributeError):
        load_config(write_toml(tmp_path, content))


def test_invalid_loss_function_name_raises(tmp_path: Path, classifier_json: Path):
    content = VALID_TOML_TEMPLATE.format(classifier_path=classifier_json).replace(
        'name = "CrossEntropyLoss"', 'name = "InvalidLoss"'
    )
    with pytest.raises(AttributeError):
        load_config(write_toml(tmp_path, content))


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent.toml"))
