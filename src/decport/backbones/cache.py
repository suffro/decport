"""Exact hidden-state caching for frozen backbones."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from decport.backbones.base import DecPortBackbone
from decport.schema import DecisionExample

CacheKey = tuple[str, str, str]


class CachedBackbone(DecPortBackbone):
    """Reuse deterministic frozen representations across experiment conditions.

    Cached tensors live on CPU and are moved to the wrapped backbone's device when
    requested. Because the wrapped backbone is frozen and permanently in eval mode,
    this is exactly equivalent to repeated extraction while avoiding redundant LM
    forward passes.
    """

    def __init__(self, backbone: DecPortBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_id = getattr(backbone, "model_id", None)
        self.max_length = getattr(backbone, "max_length", None)
        self._cache: dict[CacheKey, Tensor] = {}
        self._prompt_cache: dict[str, Tensor] = {}
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size

    @property
    def cache_size(self) -> int:
        return len(self._cache) + len(self._prompt_cache)

    def train(self, mode: bool = True) -> CachedBackbone:
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def prefill(
        self,
        examples: Iterable[DecisionExample],
        *,
        batch_size: int = 32,
    ) -> int:
        """Batch all uncached candidate representations and return additions."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        missing: list[CacheKey] = []
        seen = set(self._cache)
        for example in examples:
            for option in example.options:
                key = (example.state, example.question, option)
                if key not in seen:
                    seen.add(key)
                    missing.append(key)

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            hidden = self.backbone.encode(
                states=[key[0] for key in batch],
                questions=[key[1] for key in batch],
                options=[key[2] for key in batch],
            )
            if hidden.shape != (len(batch), self.hidden_size):
                raise RuntimeError("backbone returned an unexpected hidden-state shape")
            for key, row in zip(batch, hidden, strict=True):
                self._cache[key] = row.detach().cpu().clone()
        return len(missing)

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
        keys = list(zip(states, questions, options, strict=True))
        missing = [key for key in keys if key not in self._cache]
        if missing:
            hidden = self.backbone.encode(
                states=[key[0] for key in missing],
                questions=[key[1] for key in missing],
                options=[key[2] for key in missing],
            )
            for key, row in zip(missing, hidden, strict=True):
                self._cache[key] = row.detach().cpu().clone()
        device = next(self.backbone.parameters()).device
        return torch.stack([self._cache[key] for key in keys]).to(device)

    @torch.no_grad()
    def prefill_prompts(self, prompts: Iterable[str], *, batch_size: int = 32) -> int:
        """Batch all uncached rendered prompts, shortest first, and return additions."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        missing = sorted(
            {prompt for prompt in prompts if prompt not in self._prompt_cache},
            key=lambda prompt: (len(prompt), prompt),
        )
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            hidden = self.backbone.encode_prompts(batch)
            if hidden.shape != (len(batch), self.hidden_size):
                raise RuntimeError("backbone returned an unexpected hidden-state shape")
            for prompt, row in zip(batch, hidden, strict=True):
                self._prompt_cache[prompt] = row.detach().cpu().clone()
        return len(missing)

    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str]) -> Tensor:
        if not prompts:
            raise ValueError("cannot encode an empty batch")
        missing = list(
            dict.fromkeys(prompt for prompt in prompts if prompt not in self._prompt_cache)
        )
        if missing:
            hidden = self.backbone.encode_prompts(missing)
            for prompt, row in zip(missing, hidden, strict=True):
                self._prompt_cache[prompt] = row.detach().cpu().clone()
        device = next(self.backbone.parameters()).device
        return torch.stack([self._prompt_cache[prompt] for prompt in prompts]).to(device)
