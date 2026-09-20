"""Small, explicit training loop for the three v0.1 experiment modes."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from decport.data import shuffle_options
from decport.model import DecPort
from decport.schema import DecisionExample


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0
    shuffle_training_options: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    epoch_losses: tuple[float, ...]


def train_decision_model(
    model: DecPort,
    examples: list[DecisionExample],
    config: TrainingConfig,
    *,
    train_head: bool,
) -> TrainingHistory:
    """Train an adapter and, for source/native runs, optionally its head."""

    if not examples:
        raise ValueError("training examples cannot be empty")
    if any(example.answer is None for example in examples):
        raise ValueError("every training example must include an answer")

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    model.backbone.requires_grad_(False)
    model.adapter.requires_grad_(True)
    model.head.requires_grad_(train_head)
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_losses: list[float] = []
    for _ in range(config.epochs):
        epoch_examples = list(examples)
        rng.shuffle(epoch_examples)
        total_loss = 0.0
        for original in epoch_examples:
            example = (
                shuffle_options(original, rng) if config.shuffle_training_options else original
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model(example.state, example.question, example.options)
            target = torch.tensor([example.answer_index], device=scores.device)
            loss = nn.functional.cross_entropy(scores.unsqueeze(0), target)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach())
        epoch_losses.append(total_loss / len(epoch_examples))

    model.eval()
    return TrainingHistory(epoch_losses=tuple(epoch_losses))
