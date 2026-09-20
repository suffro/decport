"""Backbone abstractions and shared Hugging Face extraction logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class DecPortBackbone(nn.Module, ABC):
    """Minimal contract exposed to the backbone-agnostic DecPort model."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the extracted native representation."""

    @abstractmethod
    def encode(
        self,
        states: Sequence[str],
        questions: Sequence[str],
        options: Sequence[str],
    ) -> Tensor:
        """Return one frozen hidden representation per candidate option."""


class HuggingFaceCausalBackbone(DecPortBackbone):
    """Frozen last-token representation from a Hugging Face causal LM."""

    DEFAULT_MODEL_ID: ClassVar[str]

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_id: str | None = None,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not hasattr(model.config, "hidden_size"):
            raise ValueError("model config does not expose hidden_size")

        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.max_length = max_length
        self.model.requires_grad_(False)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs either a pad token or an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Preserve the candidate option and question at the end of long prompts.
        self.tokenizer.truncation_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        max_length: int = 512,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> HuggingFaceCausalBackbone:
        """Load and freeze the configured Hugging Face model."""

        selected_model_id = model_id or cls.DEFAULT_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(
            selected_model_id,
            **(tokenizer_kwargs or {}),
        )
        model = AutoModelForCausalLM.from_pretrained(
            selected_model_id,
            **(model_kwargs or {}),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=selected_model_id,
            max_length=max_length,
        )

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @staticmethod
    def format_prompt(state: str, question: str, option: str) -> str:
        """Use the same representation prompt for every backbone."""

        rendered_state = state.strip() or "(none)"
        return (
            f"State:\n{rendered_state}\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Candidate option:\n{option.strip()}\n\n"
            "Decision:"
        )

    def train(self, mode: bool = True) -> HuggingFaceCausalBackbone:
        """Keep the frozen LM deterministic even when its parent enters train mode."""

        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(
        self,
        states: Sequence[str],
        questions: Sequence[str],
        options: Sequence[str],
    ) -> Tensor:
        if not states:
            raise ValueError("cannot encode an empty batch")
        if not (len(states) == len(questions) == len(options)):
            raise ValueError("states, questions, and options must have equal lengths")

        prompts = [
            self.format_prompt(state, question, option)
            for state, question, option in zip(states, questions, options, strict=True)
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_device = self.model.get_input_embeddings().weight.device
        model_inputs = {key: value.to(input_device) for key, value in encoded.items()}
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("backbone did not return hidden states")

        hidden = outputs.hidden_states[-1]
        attention_mask = model_inputs["attention_mask"].to(dtype=torch.bool)
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        last_positions = positions.masked_fill(~attention_mask, -1).max(dim=1).values
        if torch.any(last_positions < 0):
            raise RuntimeError("tokenizer produced an empty sequence")
        batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_positions, last_positions]
