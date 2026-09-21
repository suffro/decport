"""Bounded label-free cross-backbone alignment diagnostic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from decport.adapter import BackboneAdapter
from decport.alignment import AlignmentConfig, train_latent_alignment, without_answers
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.data import load_jsonl
from decport.eval import evaluate
from decport.experiment import resolve_device
from decport.head import DecisionHead
from decport.metrics import decision_portability_ratio, parameter_size_bytes
from decport.model import DecPort
from decport.serialization import save_artifact
from decport.train import TrainingConfig, train_decision_model


@dataclass(frozen=True, slots=True)
class AlignmentExperimentConfig:
    train_path: str
    eval_path: str
    output_dir: str
    ood_path: str | None = None
    qwen_model_id: str = QwenBackbone.DEFAULT_MODEL_ID
    smollm_model_id: str = SmolLMBackbone.DEFAULT_MODEL_ID
    shared_size: int = 256
    max_length: int = 512
    device: str = "auto"
    permutation_trials: int = 3
    decision_training: TrainingConfig = field(default_factory=TrainingConfig)
    alignment_training: AlignmentConfig = field(default_factory=AlignmentConfig)


def run_alignment_experiment(
    config: AlignmentExperimentConfig,
) -> dict[str, object]:
    """Train source and controls, then align a target adapter without labels."""

    if config.shared_size <= 0:
        raise ValueError("shared_size must be positive")
    if config.decision_training.seed != config.alignment_training.seed:
        raise ValueError("decision and alignment seeds must match")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_examples = load_jsonl(config.train_path)
    eval_examples = load_jsonl(config.eval_path)
    ood_examples = load_jsonl(config.ood_path) if config.ood_path else None
    unlabeled_inputs = without_answers(train_examples)
    device = resolve_device(config.device)
    _seed(config.decision_training.seed, device)

    source_backbone = QwenBackbone.from_pretrained(
        config.qwen_model_id,
        max_length=config.max_length,
        model_kwargs={"dtype": "auto"},
    ).to(device)
    source_model = DecPort(
        source_backbone,
        BackboneAdapter(source_backbone.hidden_size, config.shared_size).to(device),
        DecisionHead(config.shared_size).to(device),
    )
    source_history = train_decision_model(
        source_model,
        train_examples,
        config.decision_training,
        train_head=True,
    )
    source_metrics = _evaluate_all(source_model, eval_examples, ood_examples, config)
    save_artifact(
        source_model,
        output_dir / "source",
        include_head=True,
        metadata={"role": "source"},
    )
    source_model.requires_grad_(False)
    source_model.eval()
    source_head_state = _state_copy(source_model.head)

    target_backbone = SmolLMBackbone.from_pretrained(
        config.smollm_model_id,
        max_length=config.max_length,
        model_kwargs={"dtype": "auto"},
    ).to(device)
    initial_adapter = BackboneAdapter(
        target_backbone.hidden_size, config.shared_size
    ).to(device)
    initial_adapter_state = _state_copy(initial_adapter)

    unaligned_model = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    unaligned_model.requires_grad_(False)
    unaligned_metrics = _evaluate_all(
        unaligned_model, eval_examples, ood_examples, config
    )
    save_artifact(
        unaligned_model,
        output_dir / "unaligned_target",
        include_head=False,
        metadata={"role": "unaligned_target", "head": "source/head.safetensors"},
    )

    native_model = _target_model(
        target_backbone, initial_adapter_state, None, config.shared_size, device
    )
    native_history = train_decision_model(
        native_model,
        train_examples,
        config.decision_training,
        train_head=True,
    )
    native_metrics = _evaluate_all(native_model, eval_examples, ood_examples, config)
    save_artifact(
        native_model,
        output_dir / "native_target",
        include_head=True,
        metadata={"role": "native_target"},
    )

    transfer_model = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    transfer_history = train_decision_model(
        transfer_model,
        train_examples,
        config.decision_training,
        train_head=False,
    )
    transfer_metrics = _evaluate_all(
        transfer_model, eval_examples, ood_examples, config
    )
    save_artifact(
        transfer_model,
        output_dir / "supervised_transfer_target",
        include_head=False,
        metadata={
            "role": "supervised_transfer_target",
            "head": "source/head.safetensors",
        },
    )

    random_head = DecisionHead(config.shared_size).to(device)
    random_head.requires_grad_(False)
    random_model = _target_model(
        target_backbone,
        initial_adapter_state,
        _state_copy(random_head),
        config.shared_size,
        device,
    )
    random_history = train_decision_model(
        random_model,
        train_examples,
        config.decision_training,
        train_head=False,
    )
    random_metrics = _evaluate_all(random_model, eval_examples, ood_examples, config)
    save_artifact(
        random_model,
        output_dir / "random_head_control",
        include_head=True,
        metadata={"role": "random_head_control", "head": "random_frozen"},
    )

    aligned_model = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    source_frozen_before = {
        "adapter": _state_copy(source_model.adapter),
        "head": _state_copy(source_model.head),
    }
    aligned_head_before = _state_copy(aligned_model.head)
    alignment_history = train_latent_alignment(
        source_model,
        aligned_model,
        unlabeled_inputs,
        config.alignment_training,
    )
    _assert_state_unchanged(source_frozen_before["adapter"], source_model.adapter)
    _assert_state_unchanged(source_frozen_before["head"], source_model.head)
    _assert_state_unchanged(aligned_head_before, aligned_model.head)
    aligned_metrics = _evaluate_all(aligned_model, eval_examples, ood_examples, config)
    save_artifact(
        aligned_model,
        output_dir / "label_free_aligned_target",
        include_head=False,
        metadata={
            "role": "label_free_aligned_target",
            "head": "source/head.safetensors",
            "decision_labels_used_for_adapter": "false",
        },
    )

    conditions = {
        "source": source_metrics,
        "native_target": native_metrics,
        "supervised_transfer_target": transfer_metrics,
        "random_head_control": random_metrics,
        "unaligned_target": unaligned_metrics,
        "label_free_aligned_target": aligned_metrics,
    }
    result: dict[str, object] = {
        "status": "diagnostic_not_accepted_benchmark",
        "config": _config_to_dict(config),
        "label_free_audit": {
            "alignment_examples": len(unlabeled_inputs),
            "alignment_examples_with_answers": sum(
                example.answer is not None for example in unlabeled_inputs
            ),
            "optimized_component": "label_free_aligned_target.adapter",
            "frozen_components": [
                "qwen_backbone",
                "qwen_adapter",
                "qwen_decision_head",
                "smollm_backbone",
                "label_free_aligned_target.decision_head",
            ],
            "loss": "cosine_plus_mse",
            "decision_loss_used": False,
        },
        "source": {
            "training": {"epoch_losses": list(source_history.epoch_losses)},
            "metrics": source_metrics,
        },
        "native_target": {
            "training": {"epoch_losses": list(native_history.epoch_losses)},
            "metrics": native_metrics,
        },
        "supervised_transfer_target": {
            "training": {"epoch_losses": list(transfer_history.epoch_losses)},
            "metrics": transfer_metrics,
        },
        "random_head_control": {
            "training": {"epoch_losses": list(random_history.epoch_losses)},
            "metrics": random_metrics,
        },
        "unaligned_target": {"metrics": unaligned_metrics},
        "label_free_aligned_target": {
            "training": {
                "epoch_metrics": [asdict(epoch) for epoch in alignment_history.epochs]
            },
            "metrics": aligned_metrics,
        },
        "comparisons": _comparisons(conditions),
        "adapter_size_bytes": parameter_size_bytes(aligned_model.adapter),
        "alignment_trainable_parameters": sum(
            parameter.numel() for parameter in aligned_model.adapter.parameters()
        ),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _target_model(
    backbone,
    adapter_state,
    head_state,
    shared_size,
    device,
) -> DecPort:
    adapter = BackboneAdapter(backbone.hidden_size, shared_size).to(device)
    adapter.load_state_dict(adapter_state)
    head = DecisionHead(shared_size).to(device)
    if head_state is not None:
        head.load_state_dict(head_state)
    return DecPort(backbone, adapter, head)


def _evaluate_all(model, eval_examples, ood_examples, config):
    result = {
        "in_distribution": evaluate(
            model,
            eval_examples,
            permutation_trials=config.permutation_trials,
            seed=config.decision_training.seed,
        ).to_dict()
    }
    if ood_examples is not None:
        result["out_of_distribution"] = evaluate(
            model,
            ood_examples,
            permutation_trials=config.permutation_trials,
            seed=config.decision_training.seed,
        ).to_dict()
    return result


def _comparisons(conditions):
    comparisons = {}
    for split in conditions["native_target"]:
        accuracy = {
            name: float(metrics[split]["accuracy"])
            for name, metrics in conditions.items()
        }
        native_accuracy = accuracy["native_target"]
        comparisons[split] = {
            "supervised_transfer_dpr": (
                decision_portability_ratio(
                    accuracy["supervised_transfer_target"], native_accuracy
                )
                if native_accuracy > 0
                else None
            ),
            "label_free_alignment_dpr": (
                decision_portability_ratio(
                    accuracy["label_free_aligned_target"], native_accuracy
                )
                if native_accuracy > 0
                else None
            ),
            "label_free_gain_over_unaligned": (
                accuracy["label_free_aligned_target"] - accuracy["unaligned_target"]
            ),
            "label_free_gain_over_random_head": (
                accuracy["label_free_aligned_target"]
                - accuracy["random_head_control"]
            ),
            "supervised_transfer_gain_over_random_head": (
                accuracy["supervised_transfer_target"]
                - accuracy["random_head_control"]
            ),
        }
    return comparisons


def _state_copy(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _assert_state_unchanged(before, module):
    after = module.state_dict()
    if before.keys() != after.keys() or any(
        not torch.equal(before[key], after[key].detach().cpu()) for key in before
    ):
        raise RuntimeError("a frozen alignment component changed during training")


def _config_to_dict(config):
    value = asdict(config)
    value["decision_training"] = asdict(config.decision_training)
    value["alignment_training"] = asdict(config.alignment_training)
    return value


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
