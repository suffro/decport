"""Strictly label-free decision-space distillation for DecPort."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from decport.model import DecPort
from decport.schema import DecisionExample


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    temperature: float = 2.0
    centered_logit_mse_weight: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.centered_logit_mse_weight < 0:
            raise ValueError("centered_logit_mse_weight cannot be negative")


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    scores: Tensor


@dataclass(frozen=True, slots=True)
class DistillationEpoch:
    epoch: int
    loss: float
    kl_divergence: float
    centered_logit_mse: float


@dataclass(frozen=True, slots=True)
class DistillationHistory:
    epochs: tuple[DistillationEpoch, ...]


@dataclass(frozen=True, slots=True)
class DecisionMatch:
    kl_divergence: float
    centered_logit_mse: float
    top_choice_agreement: float
    probability_correlation: float


def collect_teacher_outputs(
    teacher_model: DecPort,
    examples: list[DecisionExample],
) -> tuple[TeacherOutput, ...]:
    """Collect frozen teacher scores once for unlabeled decision inputs."""

    _validate_examples(examples)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    outputs: list[TeacherOutput] = []
    with torch.inference_mode():
        for example in examples:
            scores = teacher_model(
                example.state, example.question, example.options
            )
            outputs.append(TeacherOutput(scores=scores.float().cpu().clone()))
    return tuple(outputs)


def permute_teacher_outputs(
    examples: list[DecisionExample],
    outputs: tuple[TeacherOutput, ...],
    *,
    seed: int,
    group_key: Callable[[DecisionExample], Hashable] | None = None,
    keep_singletons: bool = False,
) -> tuple[TeacherOutput, ...]:
    """Derange teacher outputs within groups, by default equal-option-count groups.

    ``group_key`` can make groups stricter, for example so typed decisions only exchange
    outputs with decisions of the same kind and width. A one-member group cannot be deranged;
    it raises unless ``keep_singletons`` leaves it with its own output, which can only make the
    mismatched control closer to the correct teacher. Callers must report such fixed points.
    """

    _validate_teacher_outputs(examples, outputs)
    key = group_key or (lambda example: len(example.options))
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[key(example)].append(index)

    rng = random.Random(seed)
    permuted: list[TeacherOutput | None] = [None] * len(outputs)
    for group, indices in groups.items():
        if len(indices) == 1 and keep_singletons:
            permuted[indices[0]] = outputs[indices[0]]
            continue
        if len(indices) < 2:
            raise ValueError(
                f"permuted-teacher control needs at least two examples in group {group!r}"
            )
        rng.shuffle(indices)
        rotated = indices[1:] + indices[:1]
        for destination, source in zip(indices, rotated, strict=True):
            permuted[destination] = outputs[source]
    if any(output is None for output in permuted):
        raise RuntimeError("teacher permutation did not cover every example")
    return tuple(output for output in permuted if output is not None)


def train_decision_distillation(
    teacher_model: DecPort,
    student_model: DecPort,
    examples: list[DecisionExample],
    teacher_outputs: tuple[TeacherOutput, ...],
    config: DistillationConfig,
) -> DistillationHistory:
    """Optimize only the student adapter against fixed teacher distributions."""

    _validate_teacher_outputs(examples, teacher_outputs)
    _freeze_for_distillation(teacher_model, student_model)
    _assert_distillation_freeze(teacher_model, student_model)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    trainable = list(student_model.adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[DistillationEpoch] = []
    indexed_examples = list(enumerate(examples))
    for epoch_index in range(config.epochs):
        rng.shuffle(indexed_examples)
        total_loss = 0.0
        total_kl = 0.0
        total_mse = 0.0
        for index, example in indexed_examples:
            optimizer.zero_grad(set_to_none=True)
            student_scores = student_model(
                example.state, example.question, example.options
            ).float()
            teacher_scores = teacher_outputs[index].scores.to(student_scores.device)
            kl_divergence, centered_mse = _loss_components(
                student_scores, teacher_scores, config.temperature
            )
            loss = (
                config.temperature**2 * kl_divergence
                + config.centered_logit_mse_weight * centered_mse
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach())
            total_kl += float(kl_divergence.detach())
            total_mse += float(centered_mse.detach())

        count = len(indexed_examples)
        history.append(
            DistillationEpoch(
                epoch=epoch_index + 1,
                loss=total_loss / count,
                kl_divergence=total_kl / count,
                centered_logit_mse=total_mse / count,
            )
        )

    student_model.eval()
    _assert_distillation_freeze(teacher_model, student_model)
    return DistillationHistory(epochs=tuple(history))


def measure_decision_match(
    student_model: DecPort,
    examples: list[DecisionExample],
    teacher_outputs: tuple[TeacherOutput, ...],
    config: DistillationConfig,
) -> DecisionMatch:
    """Measure student behavior against the matching teacher outputs."""

    _validate_teacher_outputs(examples, teacher_outputs)
    student_model.eval()
    total_kl = 0.0
    total_mse = 0.0
    agreements = 0
    teacher_probabilities: list[float] = []
    student_probabilities: list[float] = []
    with torch.inference_mode():
        for example, output in zip(examples, teacher_outputs, strict=True):
            student_scores = student_model(
                example.state, example.question, example.options
            ).float()
            teacher_scores = output.scores.to(student_scores.device)
            kl_divergence, centered_mse = _loss_components(
                student_scores, teacher_scores, config.temperature
            )
            total_kl += float(kl_divergence)
            total_mse += float(centered_mse)
            agreements += int(student_scores.argmax() == teacher_scores.argmax())
            teacher_probabilities.extend(
                torch.softmax(teacher_scores / config.temperature, dim=0).cpu().tolist()
            )
            student_probabilities.extend(
                torch.softmax(student_scores / config.temperature, dim=0).cpu().tolist()
            )

    count = len(examples)
    return DecisionMatch(
        kl_divergence=total_kl / count,
        centered_logit_mse=total_mse / count,
        top_choice_agreement=agreements / count,
        probability_correlation=_pearson_correlation(
            teacher_probabilities, student_probabilities
        ),
    )


def batched_distillation_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    config: DistillationConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(loss, kl, centered_mse)`` for ``(batch, candidates)`` logits.

    The loss is ``T² × batch-mean KL(teacher || student) + w × centered-logit MSE``.
    """

    if student_scores.shape != teacher_scores.shape or student_scores.ndim != 2:
        raise ValueError("student and teacher logits must share one (batch, candidates) shape")
    teacher_probabilities = torch.softmax(teacher_scores / config.temperature, dim=1)
    student_log_probabilities = torch.log_softmax(student_scores / config.temperature, dim=1)
    kl = nn.functional.kl_div(
        student_log_probabilities, teacher_probabilities, reduction="batchmean"
    )
    centered_student = student_scores - student_scores.mean(dim=1, keepdim=True)
    centered_teacher = teacher_scores - teacher_scores.mean(dim=1, keepdim=True)
    mse = nn.functional.mse_loss(centered_student, centered_teacher)
    loss = config.temperature**2 * kl + config.centered_logit_mse_weight * mse
    return loss, kl, mse


def _loss_components(
    student_scores: Tensor,
    teacher_scores: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    teacher_probabilities = torch.softmax(teacher_scores / temperature, dim=0)
    student_log_probabilities = torch.log_softmax(
        student_scores / temperature, dim=0
    )
    kl_divergence = nn.functional.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="sum",
    )
    centered_student = student_scores - student_scores.mean()
    centered_teacher = teacher_scores - teacher_scores.mean()
    centered_mse = nn.functional.mse_loss(centered_student, centered_teacher)
    return kl_divergence, centered_mse


def _validate_examples(examples: list[DecisionExample]) -> None:
    if not examples:
        raise ValueError("distillation examples cannot be empty")
    if any(example.answer is not None for example in examples):
        raise ValueError("label-free distillation examples must not include answers")


def _validate_teacher_outputs(
    examples: list[DecisionExample],
    outputs: tuple[TeacherOutput, ...],
) -> None:
    _validate_examples(examples)
    if len(outputs) != len(examples):
        raise ValueError("teacher outputs must match distillation examples")
    for example, output in zip(examples, outputs, strict=True):
        if output.scores.ndim != 1 or len(output.scores) != len(example.options):
            raise ValueError("each teacher output must match its candidate count")
        if output.scores.requires_grad:
            raise ValueError("teacher outputs must be detached from autograd")
        if not torch.isfinite(output.scores).all():
            raise ValueError("teacher outputs must be finite")


def _freeze_for_distillation(
    teacher_model: DecPort,
    student_model: DecPort,
) -> None:
    teacher_model.requires_grad_(False)
    student_model.backbone.requires_grad_(False)
    student_model.head.requires_grad_(False)
    student_model.adapter.requires_grad_(True)
    teacher_model.eval()
    student_model.train()


def _assert_distillation_freeze(
    teacher_model: DecPort,
    student_model: DecPort,
) -> None:
    if any(parameter.requires_grad for parameter in teacher_model.parameters()):
        raise RuntimeError("teacher model must remain frozen during distillation")
    if any(parameter.requires_grad for parameter in student_model.backbone.parameters()):
        raise RuntimeError("student backbone must remain frozen during distillation")
    if any(parameter.requires_grad for parameter in student_model.head.parameters()):
        raise RuntimeError("decision head must remain frozen during distillation")
    if not all(parameter.requires_grad for parameter in student_model.adapter.parameters()):
        raise RuntimeError("only the complete student adapter may be trainable")


def _pearson_correlation(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("correlation inputs must be non-empty and equal length")
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    covariance = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    first_variance = sum((value - first_mean) ** 2 for value in first)
    second_variance = sum((value - second_mean) ** 2 for value in second)
    denominator = math.sqrt(first_variance * second_variance)
    return covariance / denominator if denominator > 0 else 0.0
