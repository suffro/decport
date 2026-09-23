"""Label-free latent alignment between frozen DecPort models."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from decport.model import DecPort
from decport.schema import DecisionExample

ALIGNMENT_METHODS = (
    "cosine_mse",
    "whitened_cosine_mse",
    "ridge",
    "orthogonal_procrustes",
)
ITERATIVE_ALIGNMENT_METHODS = ("cosine_mse", "whitened_cosine_mse")
CLOSED_FORM_ALIGNMENT_METHODS = ("ridge", "orthogonal_procrustes")


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    cosine_weight: float = 1.0
    mse_weight: float = 1.0
    whitening_epsilon: float = 1e-3
    ridge_alpha: float = 1.0
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
        if self.whitening_epsilon <= 0:
            raise ValueError("whitening_epsilon must be positive")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")


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
        DecisionExample(
            example.state,
            example.question,
            example.options,
            dataset=example.dataset,
            task_family=example.task_family,
            decision_type=example.decision_type,
        )
        for example in examples
    ]


def train_latent_alignment(
    source_model: DecPort,
    target_model: DecPort,
    examples: list[DecisionExample],
    config: AlignmentConfig,
    *,
    method: str = "cosine_mse",
) -> AlignmentHistory:
    """Match target latents to frozen source latents without decision labels.

    The function deliberately rejects labeled records. Only the target adapter is
    passed to the optimizer; both backbones, the source adapter, and both heads
    remain frozen.
    """

    if method not in ITERATIVE_ALIGNMENT_METHODS:
        raise ValueError(f"unsupported iterative alignment method: {method}")
    _validate_alignment(source_model, target_model, examples)

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    _freeze_for_alignment(source_model, target_model)
    whitening = (
        _source_whitening(source_model, examples, config.whitening_epsilon)
        if method == "whitened_cosine_mse"
        else None
    )

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

            compared_target, compared_source = _comparison_latents(
                target_latent, source_latent, whitening
            )
            cosine_loss = 1.0 - nn.functional.cosine_similarity(
                compared_target, compared_source, dim=-1
            ).mean()
            mse_loss = nn.functional.mse_loss(compared_target, compared_source)
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


def fit_closed_form_alignment(
    source_model: DecPort,
    target_model: DecPort,
    examples: list[DecisionExample],
    config: AlignmentConfig,
    *,
    method: str,
) -> AlignmentHistory:
    """Fit and fold a label-free affine mapping into the target adapter."""

    if method not in CLOSED_FORM_ALIGNMENT_METHODS:
        raise ValueError(f"unsupported closed-form alignment method: {method}")
    _validate_alignment(source_model, target_model, examples)
    _freeze_for_alignment(source_model, target_model)
    with torch.no_grad():
        target_latent, source_latent = _collect_current_latents(
            source_model, target_model, examples
        )
        target_mean = target_latent.mean(dim=0)
        source_mean = source_latent.mean(dim=0)
        centered_target = target_latent - target_mean
        centered_source = source_latent - source_mean
        cross_covariance = centered_target.T @ centered_source
        if method == "ridge":
            identity = torch.eye(
                centered_target.shape[1],
                device=centered_target.device,
                dtype=centered_target.dtype,
            )
            mapping = torch.linalg.solve(
                centered_target.T @ centered_target
                + config.ridge_alpha * identity,
                cross_covariance,
            )
        else:
            left, _, right_transpose = torch.linalg.svd(
                cross_covariance, full_matrices=False
            )
            mapping = left @ right_transpose
        intercept = source_mean - target_mean @ mapping
        mapped = target_latent @ mapping + intercept
        _fold_affine_mapping(target_model, mapping, intercept)

    target_model.eval()
    return AlignmentHistory(
        epochs=(
            _alignment_epoch(
                mapped,
                source_latent,
                config,
                epoch=1,
            ),
        )
    )


def measure_latent_alignment(
    source_model: DecPort,
    target_model: DecPort,
    examples: list[DecisionExample],
    config: AlignmentConfig,
) -> AlignmentEpoch:
    """Measure raw cosine-plus-MSE alignment without changing any component."""

    _validate_alignment(source_model, target_model, examples)
    source_model.eval()
    target_model.eval()
    with torch.no_grad():
        target_latent, source_latent = _collect_current_latents(
            source_model, target_model, examples
        )
    return _alignment_epoch(target_latent, source_latent, config, epoch=0)


def _validate_alignment(
    source_model: DecPort,
    target_model: DecPort,
    examples: list[DecisionExample],
) -> None:
    if not examples:
        raise ValueError("alignment examples cannot be empty")
    if any(example.answer is not None for example in examples):
        raise ValueError("label-free alignment examples must not include answers")
    if source_model.adapter.shared_size != target_model.adapter.shared_size:
        raise ValueError("source and target adapters must use the same shared size")


def _freeze_for_alignment(source_model: DecPort, target_model: DecPort) -> None:
    source_model.requires_grad_(False)
    target_model.backbone.requires_grad_(False)
    target_model.head.requires_grad_(False)
    target_model.adapter.requires_grad_(True)
    source_model.eval()
    target_model.train()


def _source_whitening(
    source_model: DecPort,
    examples: list[DecisionExample],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_latents: list[torch.Tensor] = []
    with torch.no_grad():
        for example in examples:
            states = [example.state] * len(example.options)
            questions = [example.question] * len(example.options)
            hidden = source_model.backbone.encode(states, questions, example.options)
            source_latents.append(source_model.adapter(hidden).float())
        source = torch.cat(source_latents)
        mean = source.mean(dim=0)
        centered = source - mean
        covariance = centered.T @ centered / max(source.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_scale = torch.rsqrt(eigenvalues.clamp_min(epsilon))
        whitening = (eigenvectors * inverse_scale.unsqueeze(0)) @ eigenvectors.T
    return mean, whitening


def _comparison_latents(
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    whitening: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target_latent.float()
    source = source_latent.float()
    if whitening is None:
        return target, source
    mean, transform = whitening
    return (target - mean) @ transform, (source - mean) @ transform


def _collect_current_latents(
    source_model: DecPort,
    target_model: DecPort,
    examples: list[DecisionExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    target_latents: list[torch.Tensor] = []
    source_latents: list[torch.Tensor] = []
    for example in examples:
        states = [example.state] * len(example.options)
        questions = [example.question] * len(example.options)
        source_hidden = source_model.backbone.encode(
            states, questions, example.options
        )
        target_hidden = target_model.backbone.encode(
            states, questions, example.options
        )
        source_latents.append(source_model.adapter(source_hidden).float())
        target_latents.append(target_model.adapter(target_hidden).float())
    return torch.cat(target_latents), torch.cat(source_latents)


def _fold_affine_mapping(
    target_model: DecPort,
    mapping: torch.Tensor,
    intercept: torch.Tensor,
) -> None:
    output = target_model.adapter.network[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("target adapter must end with a linear layer")
    output.weight.copy_(mapping.T @ output.weight.float())
    output.bias.copy_(mapping.T @ output.bias.float() + intercept)


def _alignment_epoch(
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    config: AlignmentConfig,
    *,
    epoch: int,
) -> AlignmentEpoch:
    cosine_loss = 1.0 - nn.functional.cosine_similarity(
        target_latent.float(), source_latent.float(), dim=-1
    ).mean()
    cosine_loss = cosine_loss.clamp_min(0.0)
    mse_loss = nn.functional.mse_loss(target_latent.float(), source_latent.float())
    loss = config.cosine_weight * cosine_loss + config.mse_weight * mse_loss
    return AlignmentEpoch(
        epoch=epoch,
        loss=float(loss),
        cosine_loss=float(cosine_loss),
        mse_loss=float(mse_loss),
    )
