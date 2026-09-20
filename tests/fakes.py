"""Small deterministic Transformers-shaped test doubles."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, prompts: list[str], **_: object) -> dict[str, torch.Tensor]:
        lengths = [3 + (len(prompt) % 3) for prompt in prompts]
        width = max(lengths)
        input_ids = torch.zeros((len(prompts), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeCausalLM(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask
        offsets = torch.arange(self.config.hidden_size, device=input_ids.device)
        hidden = input_ids.unsqueeze(-1).float() + offsets
        return SimpleNamespace(hidden_states=(hidden,))
