"""Dependency-free metrics for DecPort experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch import nn


def accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    _validate_equal_non_empty(predictions, targets)
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(
        targets
    )


def macro_f1(predictions: Sequence[str], targets: Sequence[str]) -> float:
    _validate_equal_non_empty(predictions, targets)
    labels = set(predictions) | set(targets)
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            prediction == label and target == label
            for prediction, target in zip(predictions, targets)
        )
        false_positive = sum(
            prediction == label and target != label
            for prediction, target in zip(predictions, targets)
        )
        false_negative = sum(
            prediction != label and target == label
            for prediction, target in zip(predictions, targets)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]],
    target_indices: Sequence[int],
    *,
    epsilon: float = 1e-12,
) -> float:
    _validate_probabilities(probabilities, target_indices)
    losses = [
        -math.log(max(float(row[target]), epsilon))
        for row, target in zip(probabilities, target_indices)
    ]
    return sum(losses) / len(losses)


def brier_score(
    probabilities: Sequence[Sequence[float]],
    target_indices: Sequence[int],
) -> float:
    """Return the multiclass Brier score (lower is better)."""

    _validate_probabilities(probabilities, target_indices)
    losses = []
    for row, target in zip(probabilities, target_indices):
        losses.append(
            sum(
                (float(probability) - (1.0 if index == target else 0.0)) ** 2
                for index, probability in enumerate(row)
            )
        )
    return sum(losses) / len(losses)


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    target_indices: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    _validate_probabilities(probabilities, target_indices)
    if bins <= 0:
        raise ValueError("bins must be positive")

    totals = [0] * bins
    confidence_sums = [0.0] * bins
    correct_sums = [0] * bins
    for row, target in zip(probabilities, target_indices):
        prediction = max(range(len(row)), key=row.__getitem__)
        confidence = float(row[prediction])
        bin_index = min(int(confidence * bins), bins - 1)
        totals[bin_index] += 1
        confidence_sums[bin_index] += confidence
        correct_sums[bin_index] += prediction == target

    sample_count = len(target_indices)
    error = 0.0
    for count, confidence_sum, correct_sum in zip(
        totals, confidence_sums, correct_sums, strict=True
    ):
        if count:
            error += (count / sample_count) * abs(correct_sum / count - confidence_sum / count)
    return error


def decision_portability_ratio(transferred_accuracy: float, native_accuracy: float) -> float:
    if native_accuracy <= 0:
        raise ValueError("native accuracy must be positive")
    if transferred_accuracy < 0:
        raise ValueError("transferred accuracy cannot be negative")
    return transferred_accuracy / native_accuracy


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_size_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def _validate_equal_non_empty(first: Sequence[object], second: Sequence[object]) -> None:
    if not first or not second:
        raise ValueError("metric inputs cannot be empty")
    if len(first) != len(second):
        raise ValueError("metric inputs must have equal lengths")


def _validate_probabilities(
    probabilities: Sequence[Sequence[float]], target_indices: Sequence[int]
) -> None:
    _validate_equal_non_empty(probabilities, target_indices)
    for row, target in zip(probabilities, target_indices):
        if not row:
            raise ValueError("probability rows cannot be empty")
        if target < 0 or target >= len(row):
            raise ValueError("target index is outside its probability row")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in row):
            raise ValueError("probabilities must be finite values between zero and one")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("each probability row must sum to one")
