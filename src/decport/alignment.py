"""Label-free latent alignment between frozen DecPort models."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from decport.model import DecPort
from decport.schema import DecisionExample


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    cosine_weight: float = 1.0
    mse_weight: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.cosine_weight < 0 or self.mse_weight < 0:
            raise ValueError("alignment loss weights cannot be negative")
        if self.cosine_weight == 0 and self.mse_weight == 0:
            raise ValueError("at least one alignment loss weight must be positive")


@dataclass(frozen=True, slots=True)
class AlignmentEpoch:
    epoch: int
    loss: float
    cosine_loss: float
    mse_loss: float


@dataclass(frozen=True, slots=True)
class AlignmentHistory:
    epochs: tuple[AlignmentEpoch, ...]


def without_answers(examples: list[DecisionExample]) -> list[DecisionExample]:
    """Return the same decision inputs with every answer removed."""

    return [
        DecisionExample(example.state, example.question, example.options)
        for example in examples
    ]


def train_latent_alignment(
    source_model: DecPort,
    target_model: DecPort,
    examples: list[DecisionExample],
    config: AlignmentConfig,
) -> AlignmentHistory:
    """Match target latents to frozen source latents without decision labels.

    The function deliberately rejects labeled records. Only the target adapter is
    passed to the optimizer; both backbones, the source adapter, and both heads
    remain frozen.
    """

    if not examples:
        raise ValueError("alignment examples cannot be empty")
    if any(example.answer is not None for example in examples):
        raise ValueError("label-free alignment examples must not include answers")
    if source_model.adapter.shared_size != target_model.adapter.shared_size:
        raise ValueError("source and target adapters must use the same shared size")

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    source_model.requires_grad_(False)
    target_model.backbone.requires_grad_(False)
    target_model.head.requires_grad_(False)
    target_model.adapter.requires_grad_(True)
    source_model.eval()
    target_model.train()

    trainable = list(target_model.adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[AlignmentEpoch] = []
    for epoch_index in range(config.epochs):
        epoch_examples = list(examples)
        rng.shuffle(epoch_examples)
        total_loss = 0.0
        total_cosine = 0.0
        total_mse = 0.0
        for example in epoch_examples:
            states = [example.state] * len(example.options)
            questions = [example.question] * len(example.options)
            # ``no_grad`` keeps this a normal tensor that can safely participate
            # as the fixed target in the target adapter's autograd graph.
            with torch.no_grad():
                source_hidden = source_model.backbone.encode(
                    states, questions, example.options
                )
                source_latent = source_model.adapter(source_hidden)
            target_hidden = target_model.backbone.encode(
                states, questions, example.options
            )
            target_latent = target_model.adapter(target_hidden)

            cosine_loss = 1.0 - nn.functional.cosine_similarity(
                target_latent.float(), source_latent.float(), dim=-1
            ).mean()
            mse_loss = nn.functional.mse_loss(
                target_latent.float(), source_latent.float()
            )
            loss = (
                config.cosine_weight * cosine_loss
                + config.mse_weight * mse_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach())
            total_cosine += float(cosine_loss.detach())
            total_mse += float(mse_loss.detach())

        count = len(epoch_examples)
        history.append(
            AlignmentEpoch(
                epoch=epoch_index + 1,
                loss=total_loss / count,
                cosine_loss=total_cosine / count,
                mse_loss=total_mse / count,
            )
        )

    target_model.eval()
    return AlignmentHistory(epochs=tuple(history))
