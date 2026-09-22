import pytest
import torch

from decport.backbones import CachedBackbone, GemmaBackbone, QwenBackbone, SmolLMBackbone
from decport.schema import DecisionExample
from tests.fakes import FakeCausalLM, FakeTokenizer


@pytest.mark.parametrize("backbone_type", [QwenBackbone, SmolLMBackbone, GemmaBackbone])
def test_backbone_is_frozen_and_extracts_last_non_padding_token(backbone_type: type) -> None:
    model = FakeCausalLM(hidden_size=4)
    backbone = backbone_type(model=model, tokenizer=FakeTokenizer())

    hidden = backbone.encode(
        states=["short", "a longer state"],
        questions=["Choose?", "Choose?"],
        options=["alpha", "beta"],
    )

    assert hidden.shape == (2, 4)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert not backbone.model.training

    prompts = [
        backbone.format_prompt("short", "Choose?", "alpha"),
        backbone.format_prompt("a longer state", "Choose?", "beta"),
    ]
    expected_last_token_ids = torch.tensor([3 + len(prompt) % 3 for prompt in prompts])
    assert torch.equal(hidden[:, 0], expected_last_token_ids.float())


def test_backbone_stays_in_eval_mode_when_parent_enters_train_mode() -> None:
    backbone = QwenBackbone(model=FakeCausalLM(hidden_size=4), tokenizer=FakeTokenizer())

    backbone.train()

    assert not backbone.training
    assert not backbone.model.training


def test_backbone_rejects_misaligned_batches() -> None:
    backbone = QwenBackbone(model=FakeCausalLM(hidden_size=4), tokenizer=FakeTokenizer())

    with pytest.raises(ValueError, match="equal lengths"):
        backbone.encode(states=["a"], questions=["q", "q"], options=["x"])


def test_cached_backbone_prefills_exact_frozen_representations(monkeypatch) -> None:
    backbone = SmolLMBackbone(FakeCausalLM(hidden_size=4), FakeTokenizer())
    cached = CachedBackbone(backbone)
    examples = [DecisionExample("state", "question", ("a", "bb"))]
    expected = backbone.encode(["state", "state"], ["question", "question"], ["a", "bb"])

    assert cached.prefill(examples, batch_size=1) == 2
    monkeypatch.setattr(
        backbone,
        "encode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prefilled backbone must not be called")
        ),
    )

    actual = cached.encode(["state", "state"], ["question", "question"], ["bb", "a"])

    assert cached.cache_size == 2
    assert torch.equal(actual, expected.flip(0))
