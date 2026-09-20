import math

import pytest
import torch
from torch import nn

from decport.metrics import (
    accuracy,
    brier_score,
    decision_portability_ratio,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
    parameter_size_bytes,
    trainable_parameter_count,
)


def test_classification_metrics() -> None:
    predictions = ["a", "b", "b"]
    targets = ["a", "a", "b"]

    assert accuracy(predictions, targets) == pytest.approx(2 / 3)
    assert macro_f1(predictions, targets) == pytest.approx(2 / 3)


def test_probabilistic_metrics() -> None:
    probabilities = [[0.8, 0.2], [0.4, 0.6]]
    targets = [0, 1]

    assert negative_log_likelihood(probabilities, targets) == pytest.approx(
        -(math.log(0.8) + math.log(0.6)) / 2
    )
    assert brier_score(probabilities, targets) == pytest.approx(0.2)
    assert expected_calibration_error(probabilities, targets, bins=5) == pytest.approx(0.3)


def test_portability_ratio_rejects_undefined_baseline() -> None:
    assert decision_portability_ratio(0.72, 0.8) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="native accuracy"):
        decision_portability_ratio(0.5, 0.0)


def test_parameter_reporting() -> None:
    module = nn.Linear(3, 2)
    module.bias.requires_grad_(False)

    assert trainable_parameter_count(module) == 6
    assert parameter_size_bytes(module) == 8 * torch.tensor(0.0).element_size()


def test_probability_validation() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        brier_score([[0.2, 0.2]], [0])
