from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import numpy as np
import pytest

from dataset import Marker
from metrics import QualityMetrics, TimeMetrics


@pytest.fixture
def perfect_metrics():
    labels = np.array([label.value for label in Marker], dtype=np.int8)
    return QualityMetrics(labels=labels, predictions=labels.copy())


@pytest.fixture
def imperfect_metrics():
    labels = np.array([label.value for label in Marker], dtype=np.int8)
    predicitons_list = [label.value for label in Marker]
    predicitons_list[2], predicitons_list[3] = predicitons_list[3], predicitons_list[2]
    predictions = np.array(predicitons_list, dtype=np.int8)
    return QualityMetrics(labels=labels, predictions=predictions)


@pytest.fixture
def time_metrics():
    return TimeMetrics(15, 4.0)


def test_accuracy_perfect(perfect_metrics: QualityMetrics):
    assert perfect_metrics.accuracy() == pytest.approx(1.0)


def test_accuracy_imperfect(imperfect_metrics: QualityMetrics):
    assert imperfect_metrics.accuracy() == pytest.approx(1 - 2 / len(Marker))


def test_avg_time_per_image(time_metrics: TimeMetrics):
    assert (
        time_metrics.avg_time_per_image()
        == time_metrics.total_time / time_metrics.images
    )


def test_macro_f1_perfect(perfect_metrics: QualityMetrics):
    assert perfect_metrics.f1_score() == pytest.approx(1.0)


def test_macro_f1_imperfect(imperfect_metrics: QualityMetrics):
    assert imperfect_metrics.f1_score() == pytest.approx(1 - 2 / len(Marker))


def test_per_class_f1(perfect_metrics: QualityMetrics):
    score = perfect_metrics.f1_score(Marker.OTHER)
    assert score == pytest.approx(1.0)


def test_per_class_recall_wrong_prediction(imperfect_metrics: QualityMetrics):
    score = imperfect_metrics.recall(Marker.CHKALOV_STAIRCASE)
    assert score == pytest.approx(0.0)


def test_per_class_precision_wrong_prediction(imperfect_metrics: QualityMetrics):
    score = imperfect_metrics.recall(Marker.CHKALOV_STAIRCASE)
    assert score == pytest.approx(0.0)


def test_confusion_matrix_shape(perfect_metrics: QualityMetrics):
    cm = perfect_metrics.confusion_matrix()
    assert cm.shape == (len(Marker), len(Marker))


def test_confusion_matrix_perfect_is_diagonal(perfect_metrics: QualityMetrics):
    cm = perfect_metrics.confusion_matrix()
    present = [(i.value, j.value) for i in Marker for j in Marker]
    for i, j in present:
        if i == j:
            assert cm[i, i] > 0
        else:
            assert cm[i, j] == 0
