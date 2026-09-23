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


class PromptTooLongError(ValueError):
    """A rendered prompt exceeds a backbone's limit and truncation is disabled."""


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

    def encode_prompts(self, prompts: Sequence[str]) -> Tensor:
        """Return one frozen hidden representation per already-rendered prompt.

        Used when an external decision core defines its own candidate prompts.
        """

        raise NotImplementedError(f"{type(self).__name__} cannot encode rendered prompts")


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
        allow_truncation: bool = True,
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
        self.allow_truncation = allow_truncation
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
        allow_truncation: bool = True,
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
            allow_truncation=allow_truncation,
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

        return self.encode_prompts(
            [
                self.format_prompt(state, question, option)
                for state, question, option in zip(states, questions, options, strict=True)
            ]
        )

    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str]) -> Tensor:
        if not prompts:
            raise ValueError("cannot encode an empty batch")
        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=self.allow_truncation,
            max_length=self.max_length if self.allow_truncation else None,
            return_tensors="pt",
        )
        if not self.allow_truncation:
            longest = int(encoded["attention_mask"].sum(dim=1).max())
            if longest > self.max_length:
                raise PromptTooLongError(
                    f"prompt has {longest} tokens, above max_length={self.max_length}; "
                    "truncation is disabled"
                )
        input_device = self.model.get_input_embeddings().weight.device
        model_inputs = {key: value.to(input_device) for key, value in encoded.items()}
        # Only hidden states are used. Keeping one logit position avoids materializing
        # vocabulary logits for every token (Gemma 3's 262k vocabulary would otherwise dominate
        # memory); hidden states are computed before the LM head and are unchanged.
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            logits_to_keep=1,
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
