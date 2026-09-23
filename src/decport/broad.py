"""Batched training and stratified metrics for broad DecPort validations.

Includes the typed path used with an external frozen decision core: candidate prompts are rendered
by the core's source system, and Choice, Noul, and Score are scored and reported separately.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from decport.data import shuffle_options
from decport.decision_core import FrozenDecisionCore, TypedDecision
from decport.distillation import (
    DistillationConfig,
    TeacherOutput,
    _validate_teacher_outputs,
    batched_distillation_loss,
)
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

RenderPrompts = Callable[[TypedDecision], Sequence[str]]


@dataclass(frozen=True, slots=True)
class BatchedTrainingHistory:
    epoch_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CoreDistillationHistory:
    epoch_losses: tuple[float, ...]
    epoch_kl_divergence: tuple[float, ...]
    epoch_centered_logit_mse: tuple[float, ...]
    gradient_audit: dict[str, int]


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
            loss, _, _ = batched_distillation_loss(student_scores, teacher_scores, config)
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


def score_core_batch(
    model: DecPort,
    decisions: Sequence[TypedDecision],
    render: RenderPrompts,
) -> Tensor:
    """Return calibrated typed logits ``(batch, answer keys)`` for one homogeneous batch."""

    core = _require_core(model)
    kind, count = decisions[0].kind, decisions[0].candidate_count
    if any((decision.kind, decision.candidate_count) != (kind, count) for decision in decisions):
        raise ValueError("a typed batch must share one decision kind and candidate count")
    prompts: list[str] = []
    for decision in decisions:
        rendered = list(render(decision))
        if len(rendered) != count:
            raise ValueError("rendered prompts do not match the decision's candidate count")
        prompts.extend(rendered)
    hidden = model.backbone.encode_prompts(prompts)
    scores = core(model.adapter(hidden)).reshape(len(decisions), count)
    return core.calibrated_logits(kind, scores)


def train_core_distillation_batched(
    student_model: DecPort,
    decisions: list[TypedDecision],
    teacher_outputs: tuple[TeacherOutput, ...],
    config: DistillationConfig,
    *,
    render: RenderPrompts,
    batch_size: int,
) -> CoreDistillationHistory:
    """Strictly label-free distillation through a frozen decision core.

    ``teacher_outputs`` are the teacher's calibrated typed logits. Only the student adapter is
    optimized; the backbone and core stay frozen, and the core is verified unchanged afterward.
    """

    _validate_teacher_outputs(decisions, teacher_outputs)
    _validate_batch_size(batch_size)
    core = _require_core(student_model)
    core_sha256 = core.state_sha256()
    student_model.backbone.requires_grad_(False)
    core.requires_grad_(False)
    student_model.adapter.requires_grad_(True)
    student_model.train()
    _assert_only_adapter_trainable(student_model)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    trainable = list(student_model.adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    indexed = list(enumerate(decisions))
    losses, kls, mses = [], [], []
    gradient_audit: dict[str, int] | None = None
    for _ in range(config.epochs):
        batches = _indexed_batches_by_option_count(indexed, batch_size, rng, key=_typed_group)
        total = total_kl = total_mse = 0.0
        count = 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            student_logits = score_core_batch(
                student_model, [decision for _, decision in batch], render
            ).float()
            teacher_logits = torch.stack(
                [teacher_outputs[index].scores for index, _ in batch]
            ).to(student_logits.device)
            loss, kl, mse = batched_distillation_loss(student_logits, teacher_logits, config)
            loss.backward()
            if gradient_audit is None:
                gradient_audit = _gradient_audit(student_model)
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            total_kl += float(kl.detach()) * len(batch)
            total_mse += float(mse.detach()) * len(batch)
            count += len(batch)
        losses.append(total / count)
        kls.append(total_kl / count)
        mses.append(total_mse / count)
    student_model.eval()
    core.assert_frozen()
    if core.state_sha256() != core_sha256:
        raise RuntimeError("the frozen decision core changed during transfer")
    if gradient_audit is None:
        raise RuntimeError("distillation performed no optimization step")
    return CoreDistillationHistory(tuple(losses), tuple(kls), tuple(mses), gradient_audit)


def predict_core_logits(
    model: DecPort,
    decisions: Sequence[TypedDecision],
    *,
    render: RenderPrompts,
    batch_size: int,
) -> list[list[float]]:
    """Calibrated typed logits for every decision, in input order."""

    _validate_batch_size(batch_size)
    grouped: dict[tuple[str, int], list[tuple[int, TypedDecision]]] = defaultdict(list)
    for index, decision in enumerate(decisions):
        grouped[_typed_group(decision)].append((index, decision))
    output: list[list[float] | None] = [None] * len(decisions)
    model.eval()
    with torch.inference_mode():
        for group in grouped.values():
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                logits = score_core_batch(model, [decision for _, decision in batch], render)
                for (index, _), row in zip(batch, logits.float().cpu().tolist(), strict=True):
                    output[index] = row
    if any(row is None for row in output):
        raise RuntimeError("prediction did not cover every decision")
    return [row for row in output if row is not None]


def evaluate_typed(
    decisions: Sequence[TypedDecision],
    calibrated_logits: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Ground-truth metrics overall, per decision type, and per dataset (evaluation only)."""

    if not decisions or any(decision.answer is None for decision in decisions):
        raise ValueError("typed evaluation requires labeled decisions")
    if any(decision.dataset is None for decision in decisions):
        raise ValueError("typed evaluation requires dataset metadata")
    if len(decisions) != len(calibrated_logits):
        raise ValueError("logit rows must match decisions")
    probabilities = [
        torch.softmax(torch.tensor(row, dtype=torch.float64), dim=0).tolist()
        for row in calibrated_logits
    ]
    result: dict[str, object] = {
        "overall": _typed_metrics(decisions, probabilities, range(len(decisions))),
        "by_decision_type": _grouped_typed(decisions, probabilities, "decision_type"),
        "by_dataset": _grouped_typed(decisions, probabilities, "dataset"),
    }
    result["macro_across_decision_types"] = _macro_metrics(result["by_decision_type"])
    return result


def typed_match(
    decisions: Sequence[TypedDecision],
    student_logits: Sequence[Sequence[float]],
    teacher_outputs: tuple[TeacherOutput, ...],
    config: DistillationConfig,
) -> dict[str, object]:
    """Student/teacher behavior matching overall, per decision type, and per dataset."""

    _validate_teacher_outputs(list(decisions), teacher_outputs)
    if len(student_logits) != len(decisions):
        raise ValueError("student logit rows must match decisions")
    result: dict[str, object] = {
        "overall": _match_metrics(student_logits, teacher_outputs, config, range(len(decisions))),
        "by_decision_type": _grouped_match(
            decisions, student_logits, teacher_outputs, config, "decision_type"
        ),
        "by_dataset": _grouped_match(decisions, student_logits, teacher_outputs, config, "dataset"),
    }
    result["macro_across_decision_types"] = _macro_metrics(result["by_decision_type"])
    return result


def _typed_metrics(decisions, probabilities, indices):
    subset = [decisions[index] for index in indices]
    rows = [probabilities[index] for index in indices]
    targets = [decision.answer_index for decision in subset]
    predicted = [max(range(len(row)), key=row.__getitem__) for row in rows]
    predicted_keys = [decision.options[index] for decision, index in zip(subset, predicted)]
    answers = [decision.answer for decision in subset]
    metrics: dict[str, int | float] = {
        "example_count": len(subset),
        "accuracy": accuracy(predicted_keys, answers),
        "macro_f1": macro_f1(predicted_keys, answers),
        "nll": negative_log_likelihood(rows, targets),
        "brier": brier_score(rows, targets),
        "ece": expected_calibration_error(rows, targets, bins=10),
    }
    score = [position for position, decision in enumerate(subset) if decision.kind == "score"]
    if score:
        metrics["score_ordinal_mae"] = statistics.mean(
            abs(predicted[position] - targets[position]) for position in score
        )
        metrics["score_expected_value_mae"] = statistics.mean(
            abs(
                sum(level * value for level, value in enumerate(rows[position]))
                - targets[position]
            )
            for position in score
        )
    noul = [position for position, decision in enumerate(subset) if decision.kind == "noul"]
    if noul:
        metrics["noul_mean_p_true"] = statistics.mean(rows[position][1] for position in noul)
        metrics["noul_predicted_true_rate"] = statistics.mean(
            float(predicted[position] == 1) for position in noul
        )
        metrics["noul_label_true_rate"] = statistics.mean(
            float(targets[position] == 1) for position in noul
        )
    return metrics


def _grouped_typed(decisions, probabilities, field):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, decision in enumerate(decisions):
        groups[getattr(decision, field)].append(index)
    return {
        name: _typed_metrics(decisions, probabilities, indices)
        for name, indices in sorted(groups.items())
    }


def _typed_group(decision: TypedDecision) -> tuple[str, int]:
    return decision.kind, len(decision.options)


def _require_core(model: DecPort) -> FrozenDecisionCore:
    if not isinstance(model.head, FrozenDecisionCore):
        raise TypeError("typed decisions require a FrozenDecisionCore head")
    return model.head


def _assert_only_adapter_trainable(model: DecPort) -> None:
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    adapter = {id(parameter) for parameter in model.adapter.parameters()}
    if trainable != adapter:
        raise RuntimeError("only the target adapter may be trainable during transfer")


def _gradient_audit(model: DecPort) -> dict[str, int]:
    """Count tensors that received gradients, by component, after one backward pass."""

    adapter = list(model.adapter.parameters())
    audit = {
        "adapter_tensors": len(adapter),
        "adapter_tensors_with_gradient": sum(parameter.grad is not None for parameter in adapter),
        "core_tensors_with_gradient": sum(
            parameter.grad is not None for parameter in model.head.parameters()
        ),
        "backbone_tensors_with_gradient": sum(
            parameter.grad is not None for parameter in model.backbone.parameters()
        ),
    }
    if audit["core_tensors_with_gradient"] or audit["backbone_tensors_with_gradient"]:
        raise RuntimeError("gradients reached a frozen component")
    if audit["adapter_tensors_with_gradient"] != audit["adapter_tensors"]:
        raise RuntimeError("gradients did not reach every adapter tensor")
    return audit


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


def _indexed_batches_by_option_count(indexed, batch_size, rng, key=None):
    group = key or (lambda example: len(example.options))
    buckets: dict[object, list[tuple[int, DecisionExample]]] = defaultdict(list)
    for item in indexed:
        buckets[group(item[1])].append(item)
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
