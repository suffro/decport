import pytest
import torch

from decport.backbones import QwenBackbone, SmolLMBackbone
from tests.fakes import FakeCausalLM, FakeTokenizer


@pytest.mark.parametrize("backbone_type", [QwenBackbone, SmolLMBackbone])
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
