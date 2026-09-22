"""Larger multi-target decision-space distillation experiment."""

from __future__ import annotations

import gc
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from decport.adapter import BackboneAdapter
from decport.alignment import without_answers
from decport.backbones import CachedBackbone, GemmaBackbone, QwenBackbone, SmolLMBackbone
from decport.data import load_jsonl
from decport.distillation import (
    DistillationConfig,
    TeacherOutput,
    collect_teacher_outputs,
    measure_decision_match,
    permute_teacher_outputs,
    train_decision_distillation,
)
from decport.eval import evaluate
from decport.experiment import resolve_device
from decport.head import DecisionHead
from decport.model import DecPort
from decport.schema import DecisionExample
from decport.serialization import save_artifact
from decport.train import TrainingConfig, train_decision_model

TARGET_BACKBONES = {
    "smollm": SmolLMBackbone,
    "gemma": GemmaBackbone,
}
TARGET_CONDITIONS = (
    "unaligned_target",
    "decision_space_distillation",
    "permuted_teacher_distillation",
    "native_target",
    "supervised_transfer_target",
    "random_head_control",
)
MATCH_METRICS = (
    "top_choice_agreement",
    "probability_correlation",
    "kl_divergence",
    "centered_logit_mse",
)
EVALUATION_METRICS = (
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece",
    "option_permutation_robustness",
    "dpr_vs_native_target",
)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    backbone_type: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name cannot be empty")
        if self.backbone_type not in TARGET_BACKBONES:
            raise ValueError(f"unsupported target backbone type: {self.backbone_type}")
        if not self.model_id:
            raise ValueError("target model_id cannot be empty")


@dataclass(frozen=True, slots=True)
class ScaleExperimentConfig:
    train_path: str
    eval_path: str
    ood_path: str
    output_dir: str
    targets: tuple[TargetSpec, ...] = (
        TargetSpec("smollm", "smollm", SmolLMBackbone.DEFAULT_MODEL_ID),
        TargetSpec("gemma", "gemma", GemmaBackbone.DEFAULT_MODEL_ID),
    )
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    qwen_model_id: str = QwenBackbone.DEFAULT_MODEL_ID
    shared_size: int = 256
    max_length: int = 512
    cache_batch_size: int = 32
    device: str = "auto"
    permutation_trials: int = 3
    decision_training: TrainingConfig = field(default_factory=lambda: TrainingConfig(epochs=5))
    distillation_training: DistillationConfig = field(
        default_factory=lambda: DistillationConfig(epochs=5)
    )


def run_scale_experiment(config: ScaleExperimentConfig) -> dict[str, object]:
    """Run all seeds and targets while sharing each seed's exact Qwen head."""

    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_examples = load_jsonl(config.train_path)
    eval_examples = load_jsonl(config.eval_path)
    ood_examples = load_jsonl(config.ood_path)
    all_examples = [*train_examples, *eval_examples, *ood_examples]
    device = resolve_device(config.device)

    _progress("loading and caching Qwen source backbone")
    source_backbone = _load_cached_backbone(
        QwenBackbone,
        config.qwen_model_id,
        config,
        device,
        all_examples,
    )
    target_backbones = {}
    for target in config.targets:
        _progress(f"loading and caching {target.name} target backbone")
        target_backbones[target.name] = _load_cached_backbone(
            TARGET_BACKBONES[target.backbone_type],
            target.model_id,
            config,
            device,
            all_examples,
        )

    results = []
    for seed in config.seeds:
        _progress(f"starting seed {seed}")
        seed_dir = output_dir / f"seed-{seed}"
        result = _run_seed(
            config,
            seed,
            seed_dir,
            source_backbone,
            target_backbones,
            train_examples,
            eval_examples,
            ood_examples,
            device,
        )
        results.append(result)
        _progress(f"completed seed {seed}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_scale_results(results)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_seed(
    config: ScaleExperimentConfig,
    seed: int,
    output_dir: Path,
    source_backbone: CachedBackbone,
    target_backbones: dict[str, CachedBackbone],
    train_examples: list[DecisionExample],
    eval_examples: list[DecisionExample],
    ood_examples: list[DecisionExample],
    device: torch.device,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_config = _training_for_seed(config.decision_training, seed)
    distillation_config = _distillation_for_seed(config.distillation_training, seed)
    _seed(seed, device)

    source_model = DecPort(
        source_backbone,
        BackboneAdapter(source_backbone.hidden_size, config.shared_size).to(device),
        DecisionHead(config.shared_size).to(device),
    )
    source_history = train_decision_model(
        source_model, train_examples, decision_config, train_head=True
    )
    source_metrics = _evaluate_all(source_model, eval_examples, ood_examples, config, seed)
    source_dir = output_dir / "source"
    save_artifact(
        source_model,
        source_dir,
        include_head=True,
        metadata={"role": "source", "shared_across_targets": "true"},
    )
    source_model.requires_grad_(False)
    source_model.eval()
    source_state = {
        "adapter": _state_copy(source_model.adapter),
        "head": _state_copy(source_model.head),
    }
    source_head_sha256 = _sha256(source_dir / "head.safetensors")

    split_examples = {
        "train": without_answers(train_examples),
        "in_distribution": without_answers(eval_examples),
        "out_of_distribution": without_answers(ood_examples),
    }
    teacher_outputs = {
        split: collect_teacher_outputs(source_model, examples)
        for split, examples in split_examples.items()
    }

    target_results = {}
    for target in config.targets:
        _progress(f"seed {seed}: running target {target.name}")
        target_results[target.name] = _run_target(
            config,
            target,
            seed,
            output_dir / "targets" / target.name,
            source_model,
            source_state,
            source_head_sha256,
            target_backbones[target.name],
            train_examples,
            eval_examples,
            ood_examples,
            split_examples,
            teacher_outputs,
            decision_config,
            distillation_config,
            device,
        )

    _assert_state_unchanged(source_state["adapter"], source_model.adapter)
    _assert_state_unchanged(source_state["head"], source_model.head)
    result: dict[str, object] = {
        "status": "diagnostic_not_accepted_benchmark",
        "seed": seed,
        "config": _config_to_dict(config, seed),
        "source": {
            "training": {"epoch_losses": list(source_history.epoch_losses)},
            "metrics": source_metrics,
            "artifact": "source",
            "head_sha256": source_head_sha256,
        },
        "targets": target_results,
        "label_free_audit": {
            "distillation_examples": len(train_examples),
            "distillation_examples_with_answers": sum(
                example.answer is not None for example in split_examples["train"]
            ),
            "ground_truth_used_for_target_distillation": False,
            "label_based_early_stopping_or_selection": False,
            "teacher_outputs_detached": all(
                not output.scores.requires_grad
                for outputs in teacher_outputs.values()
                for output in outputs
            ),
            "optimized_components": [
                f"{target.name}.decision_space_distillation.adapter" for target in config.targets
            ]
            + [f"{target.name}.permuted_teacher_distillation.adapter" for target in config.targets],
            "frozen_components": [
                "qwen_backbone",
                "qwen_adapter",
                "qwen_decision_head",
                *[f"{target.name}_backbone" for target in config.targets],
                *[
                    f"{target.name}.{condition}.qwen_decision_head"
                    for target in config.targets
                    for condition in (
                        "decision_space_distillation",
                        "permuted_teacher_distillation",
                    )
                ],
            ],
            "matched_target_adapter_initialization_within_seed": True,
            "same_qwen_head_sha256_for_all_targets": source_head_sha256,
            "source_adapter_and_head_verified_unchanged": True,
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _run_target(
    config: ScaleExperimentConfig,
    target: TargetSpec,
    seed: int,
    output_dir: Path,
    source_model: DecPort,
    source_state: dict[str, dict[str, torch.Tensor]],
    source_head_sha256: str,
    target_backbone: CachedBackbone,
    train_examples: list[DecisionExample],
    eval_examples: list[DecisionExample],
    ood_examples: list[DecisionExample],
    split_examples: dict[str, list[DecisionExample]],
    teacher_outputs: dict[str, tuple[TeacherOutput, ...]],
    decision_config: TrainingConfig,
    distillation_config: DistillationConfig,
    device: torch.device,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_adapter = BackboneAdapter(target_backbone.hidden_size, config.shared_size).to(device)
    initial_adapter_state = _state_copy(initial_adapter)
    source_head_state = source_state["head"]
    results: dict[str, dict[str, object]] = {}

    unaligned = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    unaligned.requires_grad_(False)
    results["unaligned_target"] = _record_target_condition(
        unaligned,
        "unaligned_target",
        output_dir,
        eval_examples,
        ood_examples,
        split_examples,
        teacher_outputs,
        config,
        distillation_config,
        seed,
        include_head=False,
        training=None,
        metadata={"head": "../../../source/head.safetensors"},
    )

    native = _target_model(target_backbone, initial_adapter_state, None, config.shared_size, device)
    native_history = train_decision_model(native, train_examples, decision_config, train_head=True)
    results["native_target"] = _record_target_condition(
        native,
        "native_target",
        output_dir,
        eval_examples,
        ood_examples,
        split_examples,
        teacher_outputs,
        config,
        distillation_config,
        seed,
        include_head=True,
        training={"epoch_losses": list(native_history.epoch_losses)},
    )

    transfer = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    transfer_history = train_decision_model(
        transfer, train_examples, decision_config, train_head=False
    )
    results["supervised_transfer_target"] = _record_target_condition(
        transfer,
        "supervised_transfer_target",
        output_dir,
        eval_examples,
        ood_examples,
        split_examples,
        teacher_outputs,
        config,
        distillation_config,
        seed,
        include_head=False,
        training={"epoch_losses": list(transfer_history.epoch_losses)},
        metadata={"head": "../../../source/head.safetensors"},
    )

    random_head = DecisionHead(config.shared_size).to(device)
    random_head.requires_grad_(False)
    random_control = _target_model(
        target_backbone,
        initial_adapter_state,
        _state_copy(random_head),
        config.shared_size,
        device,
    )
    random_history = train_decision_model(
        random_control, train_examples, decision_config, train_head=False
    )
    results["random_head_control"] = _record_target_condition(
        random_control,
        "random_head_control",
        output_dir,
        eval_examples,
        ood_examples,
        split_examples,
        teacher_outputs,
        config,
        distillation_config,
        seed,
        include_head=True,
        training={"epoch_losses": list(random_history.epoch_losses)},
        metadata={"head": "random_frozen"},
    )

    permuted = permute_teacher_outputs(split_examples["train"], teacher_outputs["train"], seed=seed)
    for condition, objective_outputs in (
        ("decision_space_distillation", teacher_outputs["train"]),
        ("permuted_teacher_distillation", permuted),
    ):
        distilled = _target_model(
            target_backbone,
            initial_adapter_state,
            source_head_state,
            config.shared_size,
            device,
        )
        head_before = _state_copy(distilled.head)
        history = train_decision_distillation(
            source_model,
            distilled,
            split_examples["train"],
            objective_outputs,
            distillation_config,
        )
        _assert_state_unchanged(source_state["adapter"], source_model.adapter)
        _assert_state_unchanged(source_state["head"], source_model.head)
        _assert_state_unchanged(head_before, distilled.head)
        results[condition] = _record_target_condition(
            distilled,
            condition,
            output_dir,
            eval_examples,
            ood_examples,
            split_examples,
            teacher_outputs,
            config,
            distillation_config,
            seed,
            include_head=False,
            training={"epoch_metrics": [asdict(epoch) for epoch in history.epochs]},
            metadata={
                "head": "../../../source/head.safetensors",
                "decision_labels_used_for_adapter": "false",
                "teacher_outputs": (
                    "matched" if condition == "decision_space_distillation" else "permuted"
                ),
            },
        )

    native_metrics = results["native_target"]["metrics"]
    for condition in TARGET_CONDITIONS:
        metrics = results[condition]["metrics"]
        for split in metrics:
            native_accuracy = float(native_metrics[split]["accuracy"])
            metrics[split]["dpr_vs_native_target"] = (
                float(metrics[split]["accuracy"]) / native_accuracy if native_accuracy > 0 else 0.0
            )

    comparisons = {}
    for split in ("in_distribution", "out_of_distribution"):
        distilled_accuracy = float(
            results["decision_space_distillation"]["metrics"][split]["accuracy"]
        )
        comparisons[split] = {
            "distillation_gain_over_unaligned": distilled_accuracy
            - float(results["unaligned_target"]["metrics"][split]["accuracy"]),
            "distillation_gain_over_permuted_teacher": distilled_accuracy
            - float(results["permuted_teacher_distillation"]["metrics"][split]["accuracy"]),
        }

    return {
        "backbone_type": target.backbone_type,
        "model_id": target.model_id,
        "source_head_sha256": source_head_sha256,
        "conditions": results,
        "comparisons": comparisons,
        "matched_adapter_initialization": True,
    }


def _record_target_condition(
    model: DecPort,
    condition: str,
    output_dir: Path,
    eval_examples: list[DecisionExample],
    ood_examples: list[DecisionExample],
    split_examples: dict[str, list[DecisionExample]],
    teacher_outputs: dict[str, tuple[TeacherOutput, ...]],
    config: ScaleExperimentConfig,
    distillation_config: DistillationConfig,
    seed: int,
    *,
    include_head: bool,
    training: dict[str, object] | None,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    artifact_metadata = {"role": condition, **(metadata or {})}
    save_artifact(
        model,
        output_dir / condition,
        include_head=include_head,
        metadata=artifact_metadata,
    )
    return {
        "training": training,
        "metrics": _evaluate_all(model, eval_examples, ood_examples, config, seed),
        "decision_match": {
            split: asdict(
                measure_decision_match(
                    model, split_examples[split], teacher_outputs[split], distillation_config
                )
            )
            for split in split_examples
        },
        "artifact": condition,
    }


def aggregate_scale_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-seed metrics using sample standard deviation."""

    if not results:
        raise ValueError("at least one experiment result is required")
    seeds = [int(result["seed"]) for result in results]
    target_names = tuple(results[0]["targets"])
    if any(tuple(result["targets"]) != target_names for result in results):
        raise ValueError("all seeds must contain the same targets in the same order")

    aggregate: dict[str, object] = {
        "status": "diagnostic_not_accepted_benchmark",
        "seeds": seeds,
        "seed_count": len(seeds),
        "standard_deviation": "sample (n-1)",
        "source": {"metrics": {}},
        "targets": {},
        "source_head_sha256_by_seed": {
            str(result["seed"]): result["source"]["head_sha256"] for result in results
        },
    }
    source_metrics = aggregate["source"]["metrics"]
    for split in results[0]["source"]["metrics"]:
        source_metrics[split] = {
            metric: _mean_and_std(
                [float(result["source"]["metrics"][split][metric]) for result in results]
            )
            for metric in EVALUATION_METRICS
            if metric != "dpr_vs_native_target"
        }

    for target_name in target_names:
        target_aggregate = {
            "model_id": results[0]["targets"][target_name]["model_id"],
            "metrics": {},
            "decision_match": {},
            "comparisons": {},
        }
        for condition in TARGET_CONDITIONS:
            target_aggregate["metrics"][condition] = {}
            target_aggregate["decision_match"][condition] = {}
            for split in ("in_distribution", "out_of_distribution"):
                entries = [
                    result["targets"][target_name]["conditions"][condition] for result in results
                ]
                target_aggregate["metrics"][condition][split] = {
                    metric: _mean_and_std(
                        [float(entry["metrics"][split][metric]) for entry in entries]
                    )
                    for metric in EVALUATION_METRICS
                }
            for split in ("train", "in_distribution", "out_of_distribution"):
                target_aggregate["decision_match"][condition][split] = {
                    metric: _mean_and_std(
                        [float(entry["decision_match"][split][metric]) for entry in entries]
                    )
                    for metric in MATCH_METRICS
                }
        for split in ("in_distribution", "out_of_distribution"):
            target_aggregate["comparisons"][split] = {
                comparison: _mean_and_std(
                    [
                        float(result["targets"][target_name]["comparisons"][split][comparison])
                        for result in results
                    ]
                )
                for comparison in (
                    "distillation_gain_over_unaligned",
                    "distillation_gain_over_permuted_teacher",
                )
            }
        aggregate["targets"][target_name] = target_aggregate
    return aggregate


def _load_cached_backbone(
    backbone_type,
    model_id: str,
    config: ScaleExperimentConfig,
    device: torch.device,
    examples: list[DecisionExample],
) -> CachedBackbone:
    backbone = backbone_type.from_pretrained(
        model_id,
        max_length=config.max_length,
        model_kwargs={"dtype": "auto"},
    ).to(device)
    cached = CachedBackbone(backbone)
    cached.prefill(examples, batch_size=config.cache_batch_size)
    return cached


def _target_model(backbone, adapter_state, head_state, shared_size, device) -> DecPort:
    adapter = BackboneAdapter(backbone.hidden_size, shared_size).to(device)
    adapter.load_state_dict(adapter_state)
    head = DecisionHead(shared_size).to(device)
    if head_state is not None:
        head.load_state_dict(head_state)
    return DecPort(backbone, adapter, head)


def _evaluate_all(model, eval_examples, ood_examples, config, seed):
    return {
        "in_distribution": evaluate(
            model,
            eval_examples,
            permutation_trials=config.permutation_trials,
            seed=seed,
        ).to_dict(),
        "out_of_distribution": evaluate(
            model,
            ood_examples,
            permutation_trials=config.permutation_trials,
            seed=seed,
        ).to_dict(),
    }


def _validate_config(config: ScaleExperimentConfig) -> None:
    if config.shared_size <= 0 or config.max_length <= 0:
        raise ValueError("shared_size and max_length must be positive")
    if config.cache_batch_size <= 0:
        raise ValueError("cache_batch_size must be positive")
    if not config.seeds or len(config.seeds) != len(set(config.seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if not config.targets or len({target.name for target in config.targets}) != len(config.targets):
        raise ValueError("targets must be non-empty and have unique names")
    if config.decision_training.epochs != config.distillation_training.epochs:
        raise ValueError("all conditions must use the same epoch count")
    if (
        config.decision_training.learning_rate != config.distillation_training.learning_rate
        or config.decision_training.weight_decay != config.distillation_training.weight_decay
    ):
        raise ValueError("decision and distillation optimizer hyperparameters must match")


def _training_for_seed(config: TrainingConfig, seed: int) -> TrainingConfig:
    values = asdict(config)
    values["seed"] = seed
    return TrainingConfig(**values)


def _distillation_for_seed(config: DistillationConfig, seed: int) -> DistillationConfig:
    values = asdict(config)
    values["seed"] = seed
    return DistillationConfig(**values)


def _config_to_dict(config: ScaleExperimentConfig, seed: int) -> dict[str, object]:
    return {
        "train_path": config.train_path,
        "eval_path": config.eval_path,
        "ood_path": config.ood_path,
        "output_dir": config.output_dir,
        "targets": [asdict(target) for target in config.targets],
        "seed": seed,
        "all_seeds": list(config.seeds),
        "qwen_model_id": config.qwen_model_id,
        "shared_size": config.shared_size,
        "max_length": config.max_length,
        "cache_batch_size": config.cache_batch_size,
        "device": config.device,
        "permutation_trials": config.permutation_trials,
        "decision_training": asdict(_training_for_seed(config.decision_training, seed)),
        "distillation_training": asdict(_distillation_for_seed(config.distillation_training, seed)),
    }


def _state_copy(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_state_unchanged(before, module) -> None:
    after = module.state_dict()
    if before.keys() != after.keys() or any(
        not torch.equal(before[key], after[key].detach().cpu()) for key in before
    ):
        raise RuntimeError("a frozen component changed during target training")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
