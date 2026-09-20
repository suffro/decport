"""End-to-end dynamic Choice inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from decport.adapter import BackboneAdapter
from decport.backbones.base import DecPortBackbone
from decport.head import DecisionHead
from decport.schema import BooleanResult, ChoiceResult, DecisionExample, ScoreResult

SCORE_LEVELS = ("very_low", "low", "medium", "high", "very_high")


class DecPort(nn.Module):
    """Compose one frozen backbone, one adapter, and a reusable head."""

    def __init__(
        self,
        backbone: DecPortBackbone,
        adapter: BackboneAdapter,
        head: DecisionHead,
    ) -> None:
        super().__init__()
        if backbone.hidden_size != adapter.input_size:
            raise ValueError(
                "adapter input size does not match backbone hidden size: "
                f"{adapter.input_size} != {backbone.hidden_size}"
            )
        if adapter.shared_size != head.shared_size:
            raise ValueError(
                "adapter and head shared sizes do not match: "
                f"{adapter.shared_size} != {head.shared_size}"
            )
        self.backbone = backbone
        self.adapter = adapter
        self.head = head
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def score_options(
        self,
        state: str,
        question: str,
        options: Sequence[str],
    ) -> Tensor:
        """Return differentiable candidate scores for one dynamic Choice task."""

        example = DecisionExample(
            state=state,
            question=question,
            options=tuple(options),
        )
        option_count = len(example.options)
        hidden = self.backbone.encode(
            states=[example.state] * option_count,
            questions=[example.question] * option_count,
            options=example.options,
        )
        latent = self.adapter(hidden)
        return self.head(latent)

    def forward(
        self,
        state: str,
        question: str,
        options: Sequence[str],
    ) -> Tensor:
        return self.score_options(state, question, options)

    def choice(
        self,
        state: str,
        question: str,
        options: Sequence[str],
    ) -> ChoiceResult:
        """Select one option and return JSON-compatible scores and probabilities."""

        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                scores = self.score_options(state, question, options)
                probabilities = torch.softmax(scores.float(), dim=0)
        finally:
            self.train(was_training)

        rendered_options = tuple(options)
        selected = int(probabilities.argmax().item())
        return {
            "choice": rendered_options[selected],
            "probabilities": {
                option: probability
                for option, probability in zip(
                    rendered_options,
                    probabilities.cpu().tolist(),
                    strict=True,
                )
            },
            "scores": {
                option: score
                for option, score in zip(
                    rendered_options,
                    scores.float().cpu().tolist(),
                    strict=True,
                )
            },
        }

    def boolean(self, state: str, question: str) -> BooleanResult:
        """Represent a Boolean decision as the dynamic options ``yes`` and ``no``."""

        result = self.choice(state, question, ("yes", "no"))
        return {
            "value": result["choice"] == "yes",
            "probabilities": result["probabilities"],
        }

    def score(self, state: str, question: str) -> ScoreResult:
        """Return an expected score in [0, 1] over five ordered levels."""

        result = self.choice(state, question, SCORE_LEVELS)
        denominator = len(SCORE_LEVELS) - 1
        expected = sum(
            (index / denominator) * result["probabilities"][level]
            for index, level in enumerate(SCORE_LEVELS)
        )
        return {
            "score": expected,
            "level": result["choice"],
            "probabilities": result["probabilities"],
        }
