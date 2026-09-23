"""Batched training and stratified metrics for the broad DecPort validation."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from decport.data import shuffle_options
from decport.distillation import DistillationConfig, TeacherOutput
from decport.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
)
from decport.model import DecPort
from decport.schema import DecisionExample
from decport.train import TrainingConfig


@dataclass(frozen=True, slots=True)
class BatchedTrainingHistory:
    epoch_losses: tuple[float, ...]


def train_supervised_batched(
    model: DecPort,
    examples: list[DecisionExample],
    config: TrainingConfig,
    *,
    train_head: bool,
    batch_size: int,
) -> BatchedTrainingHistory:
    """Train from labels with option-count-homogeneous decision minibatches."""

    if not examples or any(example.answer is None for example in examples):
        raise ValueError("supervised training requires labeled examples")
    _validate_batch_size(batch_size)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    model.backbone.requires_grad_(False)
    model.adapter.requires_grad_(True)
    model.head.requires_grad_(train_head)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses = []
    for _ in range(config.epochs):
        prepared = [
            shuffle_options(example, rng) if config.shuffle_training_options else example
            for example in examples
        ]
        batches = _batches_by_option_count(prepared, batch_size, rng)
        total = 0.0
        count = 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            scores = _score_batch(model, batch)
            target = torch.tensor([example.answer_index for example in batch], device=scores.device)
            loss = nn.functional.cross_entropy(scores, target)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(total / count)
    model.eval()
    return BatchedTrainingHistory(tuple(losses))


def train_distillation_batched(
    teacher_model: DecPort,
    student_model: DecPort,
    examples: list[DecisionExample],
    teacher_outputs: tuple[TeacherOutput, ...],
    config: DistillationConfig,
    *,
    batch_size: int,
) -> BatchedTrainingHistory:
    """Strict label-free minibatched distillation; only the student adapter is optimized."""

    if not examples:
        raise ValueError("distillation examples cannot be empty")
    if any(example.answer is not None for example in examples):
        raise ValueError("label-free distillation examples must not include answers")
    if len(examples) != len(teacher_outputs):
        raise ValueError("teacher output count must match the example count")
    if any(
        len(example.options) != len(output.scores)
        for example, output in zip(examples, teacher_outputs, strict=True)
    ):
        raise ValueError("teacher output widths must match decision option counts")
    _validate_batch_size(batch_size)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    student_model.backbone.requires_grad_(False)
    student_model.head.requires_grad_(False)
    student_model.adapter.requires_grad_(True)
    student_model.train()
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    trainable = list(student_model.adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    indexed = list(enumerate(examples))
    losses = []
    for _ in range(config.epochs):
        batches = _indexed_batches_by_option_count(indexed, batch_size, rng)
        total = 0.0
        count = 0
        for batch in batches:
            indices = [index for index, _ in batch]
            decisions = [example for _, example in batch]
            optimizer.zero_grad(set_to_none=True)
            student_scores = _score_batch(student_model, decisions).float()
            teacher_scores = torch.stack([teacher_outputs[index].scores for index in indices]).to(
                student_scores.device
            )
            teacher_probabilities = torch.softmax(teacher_scores / config.temperature, dim=1)
            student_log_probabilities = torch.log_softmax(
                student_scores / config.temperature, dim=1
            )
            kl = nn.functional.kl_div(
                student_log_probabilities, teacher_probabilities, reduction="batchmean"
            )
            centered_student = student_scores - student_scores.mean(dim=1, keepdim=True)
            centered_teacher = teacher_scores - teacher_scores.mean(dim=1, keepdim=True)
            mse = nn.functional.mse_loss(centered_student, centered_teacher)
            loss = config.temperature**2 * kl + config.centered_logit_mse_weight * mse
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(total / count)
    student_model.eval()
    if any(parameter.requires_grad for parameter in teacher_model.parameters()):
        raise RuntimeError("teacher changed trainability during distillation")
    if any(parameter.requires_grad for parameter in student_model.head.parameters()):
        raise RuntimeError("shared decision head was not frozen")
    return BatchedTrainingHistory(tuple(losses))


def evaluate_stratified(
    model: DecPort,
    examples: list[DecisionExample],
    *,
    batch_size: int,
    permutation_trials: int,
    seed: int,
) -> dict[str, object]:
    """Evaluate once, then report overall and metadata-stratified metrics."""

    _validate_labeled_metadata(examples)
    if permutation_trials < 0:
        raise ValueError("permutation_trials cannot be negative")
    probabilities = _predict_probabilities(model, examples, batch_size)
    predictions = [
        example.options[max(range(len(row)), key=row.__getitem__)]
        for example, row in zip(examples, probabilities, strict=True)
    ]
    robust = [0] * len(examples)
    if permutation_trials == 0:
        robust = [1] * len(examples)
    else:
        rng = random.Random(seed)
        for _ in range(permutation_trials):
            permuted = [shuffle_options(example, rng) for example in examples]
            permuted_probabilities = _predict_probabilities(model, permuted, batch_size)
            for index, (example, row) in enumerate(
                zip(permuted, permuted_probabilities, strict=True)
            ):
                prediction = example.options[max(range(len(row)), key=row.__getitem__)]
                robust[index] += prediction == predictions[index]
    robustness = [value / max(permutation_trials, 1) for value in robust]
    indices = list(range(len(examples)))
    result = {
        "overall": _evaluation_metrics(examples, probabilities, predictions, robustness, indices),
        "by_dataset": _grouped_evaluation(
            examples, probabilities, predictions, robustness, "dataset"
        ),
        "by_task_family": _grouped_evaluation(
            examples, probabilities, predictions, robustness, "task_family"
        ),
        "by_decision_type": _grouped_evaluation(
            examples, probabilities, predictions, robustness, "decision_type"
        ),
    }
    result["macro_across_task_families"] = _macro_metrics(result["by_task_family"])
    return result


def measure_match_stratified(
    model: DecPort,
    examples: list[DecisionExample],
    teacher_outputs: tuple[TeacherOutput, ...],
    config: DistillationConfig,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Report teacher/student behavior matching overall and by task metadata."""

    if any(example.answer is not None for example in examples):
        raise ValueError("decision matching requires label-free records")
    if len(examples) != len(teacher_outputs):
        raise ValueError("teacher output count must match examples")
    scores = _predict_scores(model, examples, batch_size)
    indices = list(range(len(examples)))
    result = {
        "overall": _match_metrics(scores, teacher_outputs, config, indices),
        "by_dataset": _grouped_match(examples, scores, teacher_outputs, config, "dataset"),
        "by_task_family": _grouped_match(examples, scores, teacher_outputs, config, "task_family"),
        "by_decision_type": _grouped_match(
            examples, scores, teacher_outputs, config, "decision_type"
        ),
    }
    result["macro_across_task_families"] = _macro_metrics(result["by_task_family"])
    return result


def _score_batch(model: DecPort, examples: list[DecisionExample]) -> Tensor:
    option_count = len(examples[0].options)
    if any(len(example.options) != option_count for example in examples):
        raise ValueError("a decision batch must use one option count")
    states = [example.state for example in examples for _ in example.options]
    questions = [example.question for example in examples for _ in example.options]
    options = [option for example in examples for option in example.options]
    hidden = model.backbone.encode(states, questions, options)
    return model.head(model.adapter(hidden)).reshape(len(examples), option_count)


def _predict_scores(
    model: DecPort, examples: list[DecisionExample], batch_size: int
) -> list[list[float]]:
    _validate_batch_size(batch_size)
    grouped: dict[int, list[tuple[int, DecisionExample]]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[len(example.options)].append((index, example))
    output: list[list[float] | None] = [None] * len(examples)
    model.eval()
    with torch.inference_mode():
        for group in grouped.values():
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                scores = _score_batch(model, [example for _, example in batch]).float().cpu()
                for (index, _), row in zip(batch, scores.tolist(), strict=True):
                    output[index] = row
    if any(row is None for row in output):
        raise RuntimeError("prediction did not cover every example")
    return [row for row in output if row is not None]


def _predict_probabilities(model, examples, batch_size):
    return [
        torch.softmax(torch.tensor(row), dim=0).tolist()
        for row in _predict_scores(model, examples, batch_size)
    ]


def _evaluation_metrics(examples, probabilities, predictions, robustness, indices):
    subset = [examples[index] for index in indices]
    targets = [example.answer for example in subset]
    if any(target is None for target in targets):
        raise ValueError("evaluation requires answers")
    rows = [probabilities[index] for index in indices]
    target_indices = [example.answer_index for example in subset]
    predicted = [predictions[index] for index in indices]
    metrics: dict[str, int | float] = {
        "example_count": len(indices),
        "accuracy": accuracy(predicted, targets),
        "macro_f1": macro_f1(predicted, targets),
        "nll": negative_log_likelihood(rows, target_indices),
        "brier": brier_score(rows, target_indices),
        "ece": expected_calibration_error(rows, target_indices, bins=10),
        "option_permutation_robustness": statistics.mean(robustness[index] for index in indices),
    }
    score_indices = [index for index in indices if examples[index].decision_type == "score"]
    if score_indices:
        metrics["score_ordinal_mae"] = statistics.mean(
            abs(
                max(range(len(probabilities[index])), key=probabilities[index].__getitem__)
                - examples[index].answer_index
            )
            for index in score_indices
        )
    return metrics


def _grouped_evaluation(examples, probabilities, predictions, robustness, field):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[getattr(example, field)].append(index)
    return {
        name: _evaluation_metrics(examples, probabilities, predictions, robustness, indices)
        for name, indices in sorted(groups.items())
    }


def _match_metrics(student_scores, teacher_outputs, config, indices):
    total_kl = 0.0
    total_mse = 0.0
    agreement = 0
    teacher_probabilities: list[float] = []
    student_probabilities: list[float] = []
    for index in indices:
        student = torch.tensor(student_scores[index], dtype=torch.float32)
        teacher = teacher_outputs[index].scores.float().cpu()
        teacher_probability = torch.softmax(teacher / config.temperature, dim=0)
        student_log_probability = torch.log_softmax(student / config.temperature, dim=0)
        total_kl += float(
            nn.functional.kl_div(student_log_probability, teacher_probability, reduction="sum")
        )
        centered_student = student - student.mean()
        centered_teacher = teacher - teacher.mean()
        total_mse += float(nn.functional.mse_loss(centered_student, centered_teacher))
        agreement += int(student.argmax() == teacher.argmax())
        teacher_probabilities.extend(teacher_probability.tolist())
        student_probabilities.extend(torch.softmax(student / config.temperature, dim=0).tolist())
    count = len(indices)
    return {
        "example_count": count,
        "top_choice_agreement": agreement / count,
        "probability_correlation": _correlation(teacher_probabilities, student_probabilities),
        "kl_divergence": total_kl / count,
        "centered_logit_mse": total_mse / count,
    }


def _grouped_match(examples, scores, teacher_outputs, config, field):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[getattr(example, field)].append(index)
    return {
        name: _match_metrics(scores, teacher_outputs, config, indices)
        for name, indices in sorted(groups.items())
    }


def _macro_metrics(groups):
    metric_names = set.intersection(*(set(metrics) for metrics in groups.values())) - {
        "example_count"
    }
    return {
        metric: statistics.mean(float(values[metric]) for values in groups.values())
        for metric in sorted(metric_names)
    }


def _batches_by_option_count(examples, batch_size, rng):
    indexed = list(enumerate(examples))
    return [
        [example for _, example in batch]
        for batch in _indexed_batches_by_option_count(indexed, batch_size, rng)
    ]


def _indexed_batches_by_option_count(indexed, batch_size, rng):
    buckets: dict[int, list[tuple[int, DecisionExample]]] = defaultdict(list)
    for item in indexed:
        buckets[len(item[1].options)].append(item)
    batches = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        batches.extend(
            bucket[start : start + batch_size] for start in range(0, len(bucket), batch_size)
        )
    rng.shuffle(batches)
    return batches


def _correlation(first: list[float], second: list[float]) -> float:
    first_mean = statistics.mean(first)
    second_mean = statistics.mean(second)
    numerator = sum(
        (a - first_mean) * (b - second_mean) for a, b in zip(first, second, strict=True)
    )
    first_scale = math.sqrt(sum((a - first_mean) ** 2 for a in first))
    second_scale = math.sqrt(sum((b - second_mean) ** 2 for b in second))
    denominator = first_scale * second_scale
    return numerator / denominator if denominator else 0.0


def _validate_labeled_metadata(examples):
    if not examples or any(example.answer is None for example in examples):
        raise ValueError("evaluation requires labeled examples")
    if any(
        not example.dataset or not example.task_family or not example.decision_type
        for example in examples
    ):
        raise ValueError("stratified evaluation requires complete task metadata")


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
