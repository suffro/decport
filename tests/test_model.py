import pytest
import torch

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.backbones import QwenBackbone, SmolLMBackbone
from tests.fakes import FakeCausalLM, FakeTokenizer


def make_backbone(backbone_type: type, hidden_size: int):
    return backbone_type(model=FakeCausalLM(hidden_size), tokenizer=FakeTokenizer())


def test_two_hidden_widths_use_the_same_shared_head() -> None:
    torch.manual_seed(0)
    shared_head = DecisionHead(shared_size=8)
    qwen = DecPort(
        make_backbone(QwenBackbone, hidden_size=12),
        BackboneAdapter(input_size=12, shared_size=8),
        shared_head,
    )
    smollm = DecPort(
        make_backbone(SmolLMBackbone, hidden_size=6),
        BackboneAdapter(input_size=6, shared_size=8),
        shared_head,
    )

    qwen_scores = qwen.score_options("state", "Pick one", ["a", "b", "c"])
    smollm_scores = smollm.score_options("state", "Pick one", ["a", "b", "c"])

    assert qwen.head is smollm.head
    assert qwen_scores.shape == smollm_scores.shape == (3,)
    assert qwen_scores.requires_grad
    assert smollm_scores.requires_grad


def test_choice_supports_dynamic_options_and_normalized_probabilities() -> None:
    model = DecPort(
        make_backbone(QwenBackbone, hidden_size=5),
        BackboneAdapter(input_size=5, shared_size=7),
        DecisionHead(shared_size=7),
    )

    result = model.choice(
        state="Customer says they were charged twice.",
        question="Which department should handle this?",
        options=["billing", "sales", "technical"],
    )

    assert result["choice"] in {"billing", "sales", "technical"}
    assert set(result["probabilities"]) == {"billing", "sales", "technical"}
    assert set(result["scores"]) == {"billing", "sales", "technical"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_model_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="backbone hidden size"):
        DecPort(
            make_backbone(QwenBackbone, hidden_size=5),
            BackboneAdapter(input_size=6, shared_size=7),
            DecisionHead(shared_size=7),
        )


def test_boolean_and_basic_score_are_thin_choice_wrappers() -> None:
    model = DecPort(
        make_backbone(QwenBackbone, hidden_size=5),
        BackboneAdapter(input_size=5, shared_size=7),
        DecisionHead(shared_size=7),
    )

    boolean = model.boolean("state", "Is this true?")
    score = model.score("state", "How strong is this?")

    assert set(boolean["probabilities"]) == {"yes", "no"}
    assert boolean["value"] in {True, False}
    assert 0.0 <= score["score"] <= 1.0
    assert score["level"] in {"very_low", "low", "medium", "high", "very_high"}


def test_only_adapter_and_head_receive_gradients() -> None:
    model = DecPort(
        make_backbone(QwenBackbone, hidden_size=5),
        BackboneAdapter(input_size=5, shared_size=7),
        DecisionHead(shared_size=7),
    )
    model.train()

    loss = -torch.log_softmax(model("state", "Pick one", ["a", "b"]), dim=0)[0]
    loss.backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.adapter.parameters())
    assert all(parameter.grad is not None for parameter in model.head.parameters())
