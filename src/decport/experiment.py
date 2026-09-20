"""Reproducible source, native-target, and transfer experiment orchestration."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from decport.adapter import BackboneAdapter
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.data import load_jsonl
from decport.eval import evaluate
from decport.head import DecisionHead
from decport.metrics import decision_portability_ratio, parameter_size_bytes
from decport.model import DecPort
from decport.serialization import save_artifact
from decport.train import TrainingConfig, train_decision_model


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
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
    training: TrainingConfig = field(default_factory=TrainingConfig)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run the plan's three experiments and write their artifacts and metrics."""

    if config.shared_size <= 0:
        raise ValueError("shared_size must be positive")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_examples = load_jsonl(config.train_path)
    eval_examples = load_jsonl(config.eval_path)
    ood_examples = load_jsonl(config.ood_path) if config.ood_path else None
    device = resolve_device(config.device)
    _seed_experiment(config.training.seed, device)

    source_backbone = QwenBackbone.from_pretrained(
        config.qwen_model_id,
        max_length=config.max_length,
        model_kwargs={"dtype": "auto"},
    )
    source_backbone.to(device)
    shared_head = DecisionHead(config.shared_size).to(device)
    source_model = DecPort(
        source_backbone,
        BackboneAdapter(source_backbone.hidden_size, config.shared_size).to(device),
        shared_head,
    )
    source_history = train_decision_model(
        source_model,
        train_examples,
        config.training,
        train_head=True,
    )
    source_metrics = _evaluate_all(source_model, eval_examples, ood_examples, config)
    save_artifact(
        source_model,
        output_dir / "source",
        include_head=True,
        metadata={"role": "source"},
    )
    shared_head_state = {
        key: value.detach().cpu().clone() for key, value in source_model.head.state_dict().items()
    }
    del source_model, source_backbone, shared_head
    _release_device_cache(device)

    target_backbone = SmolLMBackbone.from_pretrained(
        config.smollm_model_id,
        max_length=config.max_length,
        model_kwargs={"dtype": "auto"},
    )
    target_backbone.to(device)

    initial_target_adapter = BackboneAdapter(
        target_backbone.hidden_size, config.shared_size
    ).to(device)
    initial_target_adapter_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_target_adapter.state_dict().items()
    }
    native_model = DecPort(
        target_backbone,
        initial_target_adapter,
        DecisionHead(config.shared_size).to(device),
    )
    native_history = train_decision_model(
        native_model,
        train_examples,
        config.training,
        train_head=True,
    )
    native_metrics = _evaluate_all(native_model, eval_examples, ood_examples, config)
    save_artifact(
        native_model,
        output_dir / "native_target",
        include_head=True,
        metadata={"role": "native_target"},
    )

    frozen_shared_head = DecisionHead(config.shared_size).to(device)
    frozen_shared_head.load_state_dict(shared_head_state)
    transfer_adapter = BackboneAdapter(target_backbone.hidden_size, config.shared_size).to(device)
    transfer_adapter.load_state_dict(initial_target_adapter_state)
    transfer_model = DecPort(
        target_backbone,
        transfer_adapter,
        frozen_shared_head,
    )
    transfer_history = train_decision_model(
        transfer_model,
        train_examples,
        config.training,
        train_head=False,
    )
    transfer_metrics = _evaluate_all(transfer_model, eval_examples, ood_examples, config)
    save_artifact(
        transfer_model,
        output_dir / "transfer_target",
        include_head=False,
        metadata={"role": "transfer_target", "head": "source/head.safetensors"},
    )

    native_accuracy = float(native_metrics["in_distribution"]["accuracy"])
    transfer_accuracy = float(transfer_metrics["in_distribution"]["accuracy"])
    portability_ratio = (
        decision_portability_ratio(transfer_accuracy, native_accuracy)
        if native_accuracy > 0
        else None
    )
    result: dict[str, object] = {
        "config": _config_to_dict(config),
        "source": {
            "training": {"epoch_losses": list(source_history.epoch_losses)},
            "metrics": source_metrics,
        },
        "native_target": {
            "training": {"epoch_losses": list(native_history.epoch_losses)},
            "metrics": native_metrics,
        },
        "transfer_target": {
            "training": {"epoch_losses": list(transfer_history.epoch_losses)},
            "metrics": transfer_metrics,
        },
        "decision_portability_ratio": portability_ratio,
        "adapter_size_bytes": {
            "native_target": parameter_size_bytes(native_model.adapter),
            "transfer_target": parameter_size_bytes(transfer_model.adapter),
        },
        "trainable_parameters": {
            "source": source_model_parameter_count_from_artifact(
                output_dir / "source", include_head=True
            ),
            "native_target": sum(
                parameter.numel()
                for module in (native_model.adapter, native_model.head)
                for parameter in module.parameters()
            ),
            "transfer_target": sum(
                parameter.numel() for parameter in transfer_model.adapter.parameters()
            ),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_all(model, eval_examples, ood_examples, config):
    result = {
        "in_distribution": evaluate(
            model,
            eval_examples,
            permutation_trials=config.permutation_trials,
            seed=config.training.seed,
        ).to_dict()
    }
    if ood_examples is not None:
        result["out_of_distribution"] = evaluate(
            model,
            ood_examples,
            permutation_trials=config.permutation_trials,
            seed=config.training.seed,
        ).to_dict()
    return result


def _config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    result = asdict(config)
    result["training"] = asdict(config.training)
    return result


def _release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _seed_experiment(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def source_model_parameter_count_from_artifact(path: Path, *, include_head: bool) -> int:
    """Read parameter counts from safe tensor files after source memory is released."""

    from safetensors import safe_open

    filenames = [path / "adapter.safetensors"]
    if include_head:
        filenames.append(path / "head.safetensors")
    total = 0
    for filename in filenames:
        with safe_open(filename, framework="pt") as tensors:
            total += sum(tensors.get_tensor(key).numel() for key in tensors.keys())
    return total
