import pytest
import torch

from decport.backbones import (
    CachedBackbone,
    GemmaBackbone,
    LlamaBackbone,
    PromptTooLongError,
    QwenBackbone,
    SmolLMBackbone,
)
from decport.schema import DecisionExample
from tests.fakes import FakeCausalLM, FakeTokenizer


@pytest.mark.parametrize(
    "backbone_type", [QwenBackbone, SmolLMBackbone, GemmaBackbone, LlamaBackbone]
)
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


def test_rendered_prompts_encode_like_formatted_candidates() -> None:
    backbone = QwenBackbone(model=FakeCausalLM(hidden_size=4), tokenizer=FakeTokenizer())

    direct = backbone.encode(["state"], ["Choose?"], ["alpha"])
    rendered = backbone.encode_prompts([backbone.format_prompt("state", "Choose?", "alpha")])

    assert torch.equal(direct, rendered)


def test_disabled_truncation_rejects_over_limit_prompts() -> None:
    # FakeTokenizer produces 3 + len(prompt) % 3 tokens.
    backbone = QwenBackbone(
        model=FakeCausalLM(hidden_size=4),
        tokenizer=FakeTokenizer(),
        max_length=3,
        allow_truncation=False,
    )

    assert backbone.encode_prompts(["abc"]).shape == (1, 4)
    with pytest.raises(PromptTooLongError, match="truncation is disabled"):
        backbone.encode_prompts(["abcd"])


def test_cached_backbone_prefills_rendered_prompts(monkeypatch) -> None:
    backbone = SmolLMBackbone(FakeCausalLM(hidden_size=4), FakeTokenizer())
    cached = CachedBackbone(backbone)
    expected = backbone.encode_prompts(["one", "three"])

    assert cached.prefill_prompts(["three", "one", "one"], batch_size=1) == 2
    monkeypatch.setattr(
        backbone,
        "encode_prompts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prefilled backbone must not be called")
        ),
    )

    assert torch.equal(cached.encode_prompts(["one", "three"]), expected)
    assert cached.cache_size == 2


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
