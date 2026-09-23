from copy import deepcopy

import pytest
import torch

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.broad import evaluate_stratified, train_distillation_batched
from decport.distillation import DistillationConfig, TeacherOutput
from decport.schema import DecisionExample
from tests.fakes import FakeCausalLM, FakeTokenizer


def _models() -> tuple[DecPort, DecPort]:
    teacher = DecPort(
        QwenBackbone(FakeCausalLM(5), FakeTokenizer()),
        BackboneAdapter(5, 6),
        DecisionHead(6),
    )
    student = DecPort(
        SmolLMBackbone(FakeCausalLM(4), FakeTokenizer()),
        BackboneAdapter(4, 6),
        DecisionHead(6),
    )
    student.head.load_state_dict(teacher.head.state_dict())
    return teacher, student


def test_batched_transfer_entry_point_rejects_labels() -> None:
    teacher, student = _models()
    labeled = [DecisionExample("state", "question", ("yes", "no"), "yes")]

    with pytest.raises(ValueError, match="must not include answers"):
        train_distillation_batched(
            teacher,
            student,
            labeled,
            (TeacherOutput(torch.tensor([1.0, 0.0])),),
            DistillationConfig(epochs=1),
            batch_size=1,
        )


def test_batched_transfer_updates_only_target_adapter() -> None:
    teacher, student = _models()
    examples = [
        DecisionExample("a", "q", ("yes", "no")),
        DecisionExample("bb", "q", ("yes", "no")),
    ]
    outputs = (
        TeacherOutput(torch.tensor([1.0, -1.0])),
        TeacherOutput(torch.tensor([-1.0, 1.0])),
    )
    teacher_before = deepcopy(teacher.state_dict())
    head_before = deepcopy(student.head.state_dict())
    adapter_before = deepcopy(student.adapter.state_dict())

    train_distillation_batched(
        teacher,
        student,
        examples,
        outputs,
        DistillationConfig(epochs=1, learning_rate=1e-2),
        batch_size=2,
    )

    assert all(
        torch.equal(teacher_before[key], teacher.state_dict()[key]) for key in teacher_before
    )
    assert all(torch.equal(head_before[key], student.head.state_dict()[key]) for key in head_before)
    assert any(
        not torch.equal(adapter_before[key], student.adapter.state_dict()[key])
        for key in adapter_before
    )


def test_stratified_evaluation_reports_dataset_type_and_macro() -> None:
    _, model = _models()
    examples = [
        DecisionExample(
            "passage",
            "true?",
            ("yes", "no"),
            "yes",
            dataset="boolq",
            task_family="reading",
            decision_type="boolean",
        ),
        DecisionExample(
            "review",
            "rating?",
            ("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
            "3 stars",
            dataset="yelp",
            task_family="rating",
            decision_type="score",
        ),
    ]

    result = evaluate_stratified(model, examples, batch_size=2, permutation_trials=0, seed=0)

    assert set(result["by_dataset"]) == {"boolq", "yelp"}
    assert set(result["by_decision_type"]) == {"boolean", "score"}
    assert "accuracy" in result["macro_across_task_families"]
    assert "score_ordinal_mae" in result["by_dataset"]["yelp"]
