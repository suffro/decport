from copy import deepcopy

import pytest
import torch

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.distillation import (
    DistillationConfig,
    TeacherOutput,
    collect_teacher_outputs,
    measure_decision_match,
    permute_teacher_outputs,
    train_decision_distillation,
)
from decport.schema import DecisionExample
from tests.fakes import FakeCausalLM, FakeTokenizer


def _models() -> tuple[DecPort, DecPort]:
    teacher_backbone = QwenBackbone(FakeCausalLM(5), FakeTokenizer())
    student_backbone = SmolLMBackbone(FakeCausalLM(4), FakeTokenizer())
    return (
        DecPort(teacher_backbone, BackboneAdapter(5, 6), DecisionHead(6)),
        DecPort(student_backbone, BackboneAdapter(4, 6), DecisionHead(6)),
    )


def _unlabeled_examples() -> list[DecisionExample]:
    return [
        DecisionExample("first", "Pick", ("a", "bb")),
        DecisionExample("second", "Pick", ("a", "bb")),
        DecisionExample("third", "Pick", ("a", "bb")),
    ]


def test_distillation_rejects_decision_labels() -> None:
    teacher, student = _models()
    labeled = [DecisionExample("state", "question", ("a", "b"), "a")]

    with pytest.raises(ValueError, match="must not include answers"):
        collect_teacher_outputs(teacher, labeled)
    with pytest.raises(ValueError, match="must not include answers"):
        train_decision_distillation(
            teacher,
            student,
            labeled,
            (TeacherOutput(torch.tensor([1.0, 0.0])),),
            DistillationConfig(epochs=1),
        )


def test_distillation_updates_only_student_adapter(monkeypatch) -> None:
    teacher, student = _models()
    examples = _unlabeled_examples()
    teacher_outputs = collect_teacher_outputs(teacher, examples)
    teacher_before = deepcopy(teacher.state_dict())
    student_backbone_before = deepcopy(student.backbone.state_dict())
    student_adapter_before = deepcopy(student.adapter.state_dict())
    student_head_before = deepcopy(student.head.state_dict())

    def reject_teacher_call(*_args, **_kwargs):
        raise AssertionError("teacher must not be called after outputs are detached")

    def reject_labeled_cross_entropy(*_args, **_kwargs):
        raise AssertionError("labeled cross-entropy must not be used by distillation")

    teacher.forward = reject_teacher_call  # type: ignore[method-assign]
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", reject_labeled_cross_entropy)
    history = train_decision_distillation(
        teacher,
        student,
        examples,
        teacher_outputs,
        DistillationConfig(epochs=1, learning_rate=1e-2),
    )

    assert len(history.epochs) == 1
    # KL is non-negative; float32 rounding of a near-zero KL can be slightly negative.
    assert history.epochs[0].loss >= -1e-6
    _assert_equal(teacher_before, teacher.state_dict())
    _assert_equal(student_backbone_before, student.backbone.state_dict())
    _assert_changed(student_adapter_before, student.adapter.state_dict())
    _assert_equal(student_head_before, student.head.state_dict())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(not parameter.requires_grad for parameter in student.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in student.head.parameters())
    assert all(parameter.requires_grad for parameter in student.adapter.parameters())


def test_permuted_teacher_is_deterministic_derangement() -> None:
    examples = _unlabeled_examples()
    outputs = tuple(
        TeacherOutput(torch.tensor([float(index), -float(index)]))
        for index in range(len(examples))
    )

    first = permute_teacher_outputs(examples, outputs, seed=7)
    second = permute_teacher_outputs(examples, outputs, seed=7)

    assert all(torch.equal(left.scores, right.scores) for left, right in zip(first, second))
    assert all(
        not torch.equal(original.scores, permuted.scores)
        for original, permuted in zip(outputs, first, strict=True)
    )


def test_batched_loss_matches_the_per_decision_objective() -> None:
    from decport.distillation import _loss_components, batched_distillation_loss

    config = DistillationConfig(temperature=2.0, centered_logit_mse_weight=0.1)
    student = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -1.0]])
    teacher = torch.tensor([[1.0, 0.0, 0.5], [-2.0, 3.0, 0.0]])

    loss, kl, mse = batched_distillation_loss(student, teacher, config)
    rows = [_loss_components(s, t, config.temperature) for s, t in zip(student, teacher)]

    assert kl.item() == pytest.approx(sum(row[0].item() for row in rows) / 2)
    assert mse.item() == pytest.approx(sum(row[1].item() for row in rows) / 2)
    assert loss.item() == pytest.approx(4.0 * kl.item() + 0.1 * mse.item())


def test_decision_match_is_invariant_to_teacher_logit_offset() -> None:
    _, student = _models()
    examples = _unlabeled_examples()
    outputs = tuple(
        TeacherOutput(torch.tensor(scores))
        for scores in ([0.5, -0.5], [1.5, -0.5], [-0.5, 1.5])
    )
    shifted = tuple(TeacherOutput(output.scores + 17.0) for output in outputs)
    config = DistillationConfig(epochs=1)

    baseline = measure_decision_match(student, examples, outputs, config)
    offset = measure_decision_match(student, examples, shifted, config)

    # Softmax is offset-invariant up to float32 KL roundoff.
    assert offset.kl_divergence == pytest.approx(baseline.kl_divergence, abs=1e-7)
    assert offset.centered_logit_mse == pytest.approx(baseline.centered_logit_mse)
    assert offset.top_choice_agreement == baseline.top_choice_agreement


def _assert_equal(before, after) -> None:
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def _assert_changed(before, after) -> None:
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[key], after[key]) for key in before)
