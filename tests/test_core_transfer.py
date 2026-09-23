from copy import deepcopy

import pytest
import torch

from decport import BackboneAdapter, DecPort
from decport.backbones import SmolLMBackbone
from decport.broad import (
    _assert_only_adapter_trainable,
    evaluate_typed,
    predict_core_logits,
    score_core_batch,
    train_core_distillation_batched,
    typed_match,
)
from decport.decision_core import LinearDecisionCore, TypedDecision
from decport.distillation import DistillationConfig, TeacherOutput, permute_teacher_outputs
from tests.fakes import FakeCausalLM, FakeTokenizer

LEVELS = ("low", "mid", "high")


def render(decision: TypedDecision) -> list[str]:
    if decision.kind == "noul":
        return [f"{decision.state}|{decision.instructions}|yes?"]
    names = decision.criteria if decision.kind == "score" else decision.options
    return [f"{decision.state}|{decision.instructions}|{name}" for name in names]


def decisions(labeled: bool = False) -> list[TypedDecision]:
    rows = []
    for index in range(2):
        rows += [
            TypedDecision(
                f"s{index}", "choice", "Pick", {"a": None, "bb": None, "ccc": None},
                answer="a" if labeled else None, dataset="arc",
            ),
            TypedDecision(
                f"p{index}", "noul", "True?",
                answer=("true", "false")[index] if labeled else None, dataset="boolq",
            ),
            TypedDecision(
                f"r{index}", "score", "Rate", LEVELS,
                answer="2" if labeled else None, dataset="yelp",
            ),
        ]
    return rows


def teacher_outputs(items: list[TypedDecision]) -> tuple[TeacherOutput, ...]:
    outputs = []
    for index, item in enumerate(items):
        if item.kind == "noul":
            outputs.append(TeacherOutput(torch.tensor([0.0, 1.5 - index])))
        else:
            outputs.append(TeacherOutput(torch.linspace(1.0, -1.0, len(item.options)) * index))
    return tuple(outputs)


def student(width: int = 6) -> DecPort:
    torch.manual_seed(0)
    core = LinearDecisionCore(width, 1.5, weight=torch.randn((1, width)), bias=torch.zeros(1))
    backbone = SmolLMBackbone(FakeCausalLM(4), FakeTokenizer())
    return DecPort(backbone, BackboneAdapter(4, width, hidden_size=3), core)


def test_core_batch_returns_calibrated_typed_logits_per_kind() -> None:
    model = student()
    items = decisions()

    choice = score_core_batch(model, [items[0], items[3]], render)
    noul = score_core_batch(model, [items[1], items[4]], render)

    assert choice.shape == (2, 3)
    assert noul.shape == (2, 2) and torch.equal(noul[:, 0], torch.zeros(2))
    with pytest.raises(ValueError, match="one decision kind"):
        score_core_batch(model, [items[0], items[1]], render)


def test_core_transfer_entry_point_rejects_labeled_decisions() -> None:
    model = student()
    labeled = decisions(labeled=True)

    with pytest.raises(ValueError, match="must not include answers"):
        train_core_distillation_batched(
            model,
            labeled,
            teacher_outputs(labeled),
            DistillationConfig(epochs=1),
            render=render,
            batch_size=2,
        )


def test_core_transfer_updates_only_the_target_adapter(monkeypatch) -> None:
    model = student()
    items = decisions()
    core_before = model.head.state_sha256()
    backbone_before = deepcopy(model.backbone.state_dict())
    adapter_before = deepcopy(model.adapter.state_dict())

    def reject_cross_entropy(*_args, **_kwargs):
        raise AssertionError("labeled cross-entropy must not be used by transfer")

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", reject_cross_entropy)
    history = train_core_distillation_batched(
        model,
        items,
        teacher_outputs(items),
        DistillationConfig(epochs=2, learning_rate=1e-2, temperature=1.0),
        render=render,
        batch_size=2,
    )

    assert len(history.epoch_losses) == 2
    assert history.gradient_audit == {
        "adapter_tensors": 6,
        "adapter_tensors_with_gradient": 6,
        "core_tensors_with_gradient": 0,
        "backbone_tensors_with_gradient": 0,
    }
    assert model.head.state_sha256() == core_before
    backbone_after = model.backbone.state_dict()
    adapter_after = model.adapter.state_dict()
    assert all(torch.equal(value, backbone_after[key]) for key, value in backbone_before.items())
    assert any(not torch.equal(value, adapter_after[key]) for key, value in adapter_before.items())
    model.head.assert_frozen()


def test_trainability_guard_rejects_anything_but_the_adapter() -> None:
    model = student()
    model.adapter.requires_grad_(True)
    _assert_only_adapter_trainable(model)

    model.head.readout.requires_grad_(True)
    with pytest.raises(RuntimeError, match="only the target adapter"):
        _assert_only_adapter_trainable(model)


def test_mismatched_teacher_is_deterministic_and_stays_within_kind() -> None:
    items = decisions() * 2
    outputs = tuple(TeacherOutput(torch.tensor([float(index)] * len(item.options)))
                    for index, item in enumerate(items))

    def key(item):
        return item.kind, len(item.options)

    first = permute_teacher_outputs(items, outputs, seed=5, group_key=key)
    second = permute_teacher_outputs(items, outputs, seed=5, group_key=key)

    assert all(torch.equal(a.scores, b.scores) for a, b in zip(first, second, strict=True))
    for item, original, permuted in zip(items, outputs, first, strict=True):
        assert not torch.equal(original.scores, permuted.scores)
        source = int(permuted.scores[0])
        assert items[source].kind == item.kind


def test_mismatched_teacher_keeps_only_an_explicit_singleton_group() -> None:
    items = decisions() * 2
    items.append(TypedDecision(
        "wide", "choice", "Pick", {"a": None, "bb": None, "ccc": None, "dddd": None},
        dataset="arc",
    ))
    outputs = tuple(TeacherOutput(torch.tensor([float(index)] * len(item.options)))
                    for index, item in enumerate(items))

    def key(item):
        return item.kind, len(item.options)

    with pytest.raises(ValueError, match="at least two examples"):
        permute_teacher_outputs(items, outputs, seed=5, group_key=key)
    permuted = permute_teacher_outputs(items, outputs, seed=5, group_key=key, keep_singletons=True)

    assert permuted[-1] is outputs[-1]
    assert all(control is not original
               for original, control in zip(outputs[:-1], permuted[:-1], strict=True))


def test_typed_evaluation_and_match_report_each_decision_type() -> None:
    model = student()
    labeled = decisions(labeled=True)
    unlabeled = [item.without_answer() for item in labeled]
    logits = predict_core_logits(model, unlabeled, render=render, batch_size=4)

    evaluation = evaluate_typed(labeled, logits)
    self_match = typed_match(
        unlabeled,
        logits,
        tuple(TeacherOutput(torch.tensor(row)) for row in logits),
        DistillationConfig(temperature=1.0),
    )

    assert set(evaluation["by_decision_type"]) == {"choice", "noul", "score"}
    assert set(evaluation["by_dataset"]) == {"arc", "boolq", "yelp"}
    assert "noul_predicted_true_rate" in evaluation["by_decision_type"]["noul"]
    assert "score_expected_value_mae" in evaluation["by_decision_type"]["score"]
    assert "accuracy" in evaluation["macro_across_decision_types"]
    assert self_match["overall"]["top_choice_agreement"] == 1.0
    assert self_match["overall"]["kl_divergence"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="labeled decisions"):
        evaluate_typed(unlabeled, logits)
