"""Evaluation loop for dynamic-choice DecPort models."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch

from decport.data import shuffle_options
from decport.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
)
from decport.model import DecPort
from decport.schema import DecisionExample


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    example_count: int
    accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece: float
    option_permutation_robustness: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate(
    model: DecPort,
    examples: list[DecisionExample],
    *,
    calibration_bins: int = 10,
    permutation_trials: int = 3,
    seed: int = 0,
) -> EvaluationResult:
    if not examples:
        raise ValueError("evaluation examples cannot be empty")
    if any(example.answer is None for example in examples):
        raise ValueError("every evaluation example must include an answer")
    if permutation_trials < 0:
        raise ValueError("permutation_trials cannot be negative")

    was_training = model.training
    model.eval()
    probabilities: list[list[float]] = []
    target_indices: list[int] = []
    predictions: list[str] = []
    targets: list[str] = []
    baseline_predictions: list[str] = []
    try:
        with torch.inference_mode():
            for example in examples:
                scores = model(example.state, example.question, example.options)
                row = torch.softmax(scores.float(), dim=0).cpu().tolist()
                probabilities.append(row)
                target_indices.append(example.answer_index)
                prediction = example.options[max(range(len(row)), key=row.__getitem__)]
                predictions.append(prediction)
                baseline_predictions.append(prediction)
                assert example.answer is not None
                targets.append(example.answer)

            robustness = _permutation_robustness(
                model,
                examples,
                baseline_predictions,
                trials=permutation_trials,
                seed=seed,
            )
    finally:
        model.train(was_training)

    return EvaluationResult(
        example_count=len(examples),
        accuracy=accuracy(predictions, targets),
        macro_f1=macro_f1(predictions, targets),
        nll=negative_log_likelihood(probabilities, target_indices),
        brier=brier_score(probabilities, target_indices),
        ece=expected_calibration_error(
            probabilities,
            target_indices,
            bins=calibration_bins,
        ),
        option_permutation_robustness=robustness,
    )


def _permutation_robustness(
    model: DecPort,
    examples: list[DecisionExample],
    baseline_predictions: list[str],
    *,
    trials: int,
    seed: int,
) -> float:
    if trials == 0:
        return 1.0
    rng = random.Random(seed)
    consistent = 0
    total = 0
    for example, baseline in zip(examples, baseline_predictions, strict=True):
        for _ in range(trials):
            permuted = _different_option_order(example, rng)
            scores = model(permuted.state, permuted.question, permuted.options)
            prediction = permuted.options[int(scores.argmax().item())]
            consistent += prediction == baseline
            total += 1
    return consistent / total


def _different_option_order(
    example: DecisionExample,
    rng: random.Random,
) -> DecisionExample:
    for _ in range(5):
        permuted = shuffle_options(example, rng)
        if permuted.options != example.options:
            return permuted
    rotated = example.options[1:] + example.options[:1]
    return DecisionExample(example.state, example.question, rotated, example.answer)
