from copy import deepcopy

import pytest
import torch

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.alignment import (
    ALIGNMENT_METHODS,
    CLOSED_FORM_ALIGNMENT_METHODS,
    AlignmentConfig,
    fit_closed_form_alignment,
    train_latent_alignment,
    without_answers,
)
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.schema import DecisionExample
from tests.fakes import FakeCausalLM, FakeTokenizer


def _models() -> tuple[DecPort, DecPort]:
    source_backbone = QwenBackbone(FakeCausalLM(5), FakeTokenizer())
    target_backbone = SmolLMBackbone(FakeCausalLM(4), FakeTokenizer())
    return (
        DecPort(source_backbone, BackboneAdapter(5, 6), DecisionHead(6)),
        DecPort(target_backbone, BackboneAdapter(4, 6), DecisionHead(6)),
    )


@pytest.mark.parametrize("method", ALIGNMENT_METHODS)
def test_alignment_rejects_decision_labels(method: str) -> None:
    source, target = _models()
    labeled = [DecisionExample("state", "question", ("a", "b"), "a")]

    with pytest.raises(ValueError, match="must not include answers"):
        _align(source, target, labeled, method)


@pytest.mark.parametrize("method", ALIGNMENT_METHODS)
def test_alignment_updates_only_target_adapter_and_does_not_call_head(
    method: str,
) -> None:
    source, target = _models()
    examples = without_answers(
        [DecisionExample("state", "question", ("a", "bb"), "a")]
    )
    source_before = deepcopy(source.state_dict())
    target_backbone_before = deepcopy(target.backbone.state_dict())
    target_adapter_before = deepcopy(target.adapter.state_dict())
    target_head_before = deepcopy(target.head.state_dict())

    def reject_head_call(*_args, **_kwargs):
        raise AssertionError("decision head must not be used by alignment loss")

    target.head.forward = reject_head_call  # type: ignore[method-assign]
    source.head.forward = reject_head_call  # type: ignore[method-assign]
    history = _align(source, target, examples, method)

    assert len(history.epochs) == 1
    assert all(epoch.loss >= 0 for epoch in history.epochs)
    _assert_equal(source_before, source.state_dict())
    _assert_equal(target_backbone_before, target.backbone.state_dict())
    _assert_changed(target_adapter_before, target.adapter.state_dict())
    _assert_equal(target_head_before, target.head.state_dict())
    assert all(not parameter.requires_grad for parameter in source.parameters())
    assert all(not parameter.requires_grad for parameter in target.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in target.head.parameters())
    assert all(parameter.requires_grad for parameter in target.adapter.parameters())


def _align(source, target, examples, method):
    config = AlignmentConfig(epochs=1, learning_rate=1e-2)
    if method in CLOSED_FORM_ALIGNMENT_METHODS:
        return fit_closed_form_alignment(
            source, target, examples, config, method=method
        )
    return train_latent_alignment(
        source, target, examples, config, method=method
    )


def _assert_equal(before, after) -> None:
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def _assert_changed(before, after) -> None:
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[key], after[key]) for key in before)
