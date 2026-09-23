"""Broad three-backbone, three-decision-type DecPort validation."""

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
from decport.backbones import (
    CachedBackbone,
    GemmaBackbone,
    LlamaBackbone,
    QwenBackbone,
    SmolLMBackbone,
)
from decport.broad import (
    evaluate_stratified,
    measure_match_stratified,
    train_distillation_batched,
    train_supervised_batched,
)
from decport.data import load_jsonl
from decport.distillation import (
    DistillationConfig,
    collect_teacher_outputs,
    permute_teacher_outputs,
)
from decport.experiment import resolve_device
from decport.head import DecisionHead
from decport.model import DecPort
from decport.serialization import save_artifact
from decport.train import TrainingConfig

TARGET_BACKBONES = {
    "smollm": SmolLMBackbone,
    "gemma": GemmaBackbone,
    "llama": LlamaBackbone,
}
TARGET_CONDITIONS = (
    "untrained_target_adapter",
    "decision_space_distillation",
    "mismatched_teacher_distillation",
    "native_target",
    "supervised_decport_target",
    "random_head_control",
)


@dataclass(frozen=True, slots=True)
class BroadTargetSpec:
    name: str
    backbone_type: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.name or self.backbone_type not in TARGET_BACKBONES or not self.model_id:
            raise ValueError("invalid broad target specification")


@dataclass(frozen=True, slots=True)
class BroadExperimentConfig:
    train_path: str
    eval_path: str
    ood_path: str
    output_dir: str
    targets: tuple[BroadTargetSpec, ...] = (
        BroadTargetSpec("smollm", "smollm", SmolLMBackbone.DEFAULT_MODEL_ID),
        BroadTargetSpec("gemma", "gemma", GemmaBackbone.DEFAULT_MODEL_ID),
        BroadTargetSpec("tinyllama", "llama", LlamaBackbone.DEFAULT_MODEL_ID),
    )
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    qwen_model_id: str = QwenBackbone.DEFAULT_MODEL_ID
    shared_size: int = 256
    max_length: int = 384
    cache_batch_size: int = 8
    train_batch_size: int = 8
    eval_batch_size: int = 32
    device: str = "auto"
    permutation_trials: int = 1
    decision_training: TrainingConfig = field(default_factory=lambda: TrainingConfig(epochs=10))
    distillation_training: DistillationConfig = field(
        default_factory=lambda: DistillationConfig(epochs=10)
    )


def run_broad_experiment(config: BroadExperimentConfig) -> dict[str, object]:
    """Run the fixed broad protocol and write all per-seed and aggregate records."""

    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_examples = load_jsonl(config.train_path)
    eval_examples = load_jsonl(config.eval_path)
    ood_examples = load_jsonl(config.ood_path)
    _validate_dataset_contract(train_examples, eval_examples, ood_examples)
    all_examples = [*train_examples, *eval_examples, *ood_examples]
    device = resolve_device(config.device)

    _progress("loading, caching, and CPU-offloading Qwen source backbone")
    source_backbone = _load_cached_backbone(
        QwenBackbone, config.qwen_model_id, config, device, all_examples
    )
    target_backbones: dict[str, CachedBackbone] = {}
    for target in config.targets:
        _progress(f"loading, caching, and CPU-offloading {target.name}")
        target_backbones[target.name] = _load_cached_backbone(
            TARGET_BACKBONES[target.backbone_type],
            target.model_id,
            config,
            device,
            all_examples,
        )

    results = []
    for seed in config.seeds:
        _progress(f"starting broad seed {seed}")
        result = _run_seed(
            config,
            seed,
            output_dir / f"seed-{seed}",
            source_backbone,
            target_backbones,
            train_examples,
            eval_examples,
            ood_examples,
            device,
        )
        results.append(result)
        _progress(f"completed broad seed {seed}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_broad_results(results)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _run_seed(
    config,
    seed,
    output_dir,
    source_backbone,
    target_backbones,
    train_examples,
    eval_examples,
    ood_examples,
    device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_config = _training_for_seed(config.decision_training, seed)
    distillation_config = _distillation_for_seed(config.distillation_training, seed)
    _seed(seed, device)
    source_model = DecPort(
        source_backbone,
        BackboneAdapter(source_backbone.hidden_size, config.shared_size).to(device),
        DecisionHead(config.shared_size).to(device),
    )
    source_history = train_supervised_batched(
        source_model,
        train_examples,
        decision_config,
        train_head=True,
        batch_size=config.train_batch_size,
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

    targets = {}
    for target in config.targets:
        _progress(f"seed {seed}: target {target.name}")
        targets[target.name] = _run_target(
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
    result = {
        "status": "broad_diagnostic_pending_interpretation",
        "seed": seed,
        "config": _config_to_dict(config, seed),
        "source": {
            "training": {"epoch_losses": list(source_history.epoch_losses)},
            "metrics": source_metrics,
            "artifact": "source",
            "head_sha256": source_head_sha256,
        },
        "targets": targets,
        "label_free_audit": {
            "transfer_entry_point_rejects_labeled_records": True,
            "distillation_examples": len(split_examples["train"]),
            "distillation_examples_with_answers": sum(
                example.answer is not None for example in split_examples["train"]
            ),
            "ground_truth_used_for_target_distillation": False,
            "label_based_early_stopping_selection_weighting_or_tuning": False,
            "teacher_outputs_detached": all(
                not output.scores.requires_grad
                for outputs in teacher_outputs.values()
                for output in outputs
            ),
            "optimized_components": [
                f"{target.name}.{condition}.adapter"
                for target in config.targets
                for condition in (
                    "decision_space_distillation",
                    "mismatched_teacher_distillation",
                )
            ],
            "frozen_components": [
                "qwen_backbone",
                "qwen_adapter",
                "qwen_decision_head",
                *[f"{target.name}_backbone" for target in config.targets],
            ],
            "same_qwen_head_sha256_for_all_targets": source_head_sha256,
            "source_adapter_and_head_verified_unchanged": True,
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _run_target(
    config,
    target,
    seed,
    output_dir,
    source_model,
    source_state,
    source_head_sha256,
    target_backbone,
    train_examples,
    eval_examples,
    ood_examples,
    split_examples,
    teacher_outputs,
    decision_config,
    distillation_config,
    device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_adapter = BackboneAdapter(target_backbone.hidden_size, config.shared_size).to(device)
    initial_adapter_state = _state_copy(initial_adapter)
    initialization_sha256 = _state_sha256(initial_adapter_state)
    source_head_state = source_state["head"]
    results = {}

    untrained = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    untrained.requires_grad_(False)
    results["untrained_target_adapter"] = _record_condition(
        untrained,
        "untrained_target_adapter",
        output_dir,
        eval_examples,
        ood_examples,
        split_examples,
        teacher_outputs,
        config,
        distillation_config,
        seed,
        False,
        None,
        {"head": "../../../source/head.safetensors"},
    )

    native = _target_model(target_backbone, initial_adapter_state, None, config.shared_size, device)
    history = train_supervised_batched(
        native,
        train_examples,
        decision_config,
        train_head=True,
        batch_size=config.train_batch_size,
    )
    results["native_target"] = _record_condition(
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
        True,
        {"epoch_losses": list(history.epoch_losses)},
    )

    supervised = _target_model(
        target_backbone, initial_adapter_state, source_head_state, config.shared_size, device
    )
    history = train_supervised_batched(
        supervised,
        train_examples,
        decision_config,
        train_head=False,
        batch_size=config.train_batch_size,
    )
    results["supervised_decport_target"] = _record_condition(
        supervised,
        "supervised_decport_target",
        output_dir,
        eval_examples,
        ood_examples,
        split_examples,
        teacher_outputs,
        config,
        distillation_config,
        seed,
        False,
        {"epoch_losses": list(history.epoch_losses)},
        {"head": "../../../source/head.safetensors"},
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
    history = train_supervised_batched(
        random_control,
        train_examples,
        decision_config,
        train_head=False,
        batch_size=config.train_batch_size,
    )
    results["random_head_control"] = _record_condition(
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
        True,
        {"epoch_losses": list(history.epoch_losses)},
        {"head": "random_frozen"},
    )

    mismatched = permute_teacher_outputs(
        split_examples["train"], teacher_outputs["train"], seed=seed
    )
    for condition, objective_outputs, teacher_kind in (
        ("decision_space_distillation", teacher_outputs["train"], "matched"),
        ("mismatched_teacher_distillation", mismatched, "deliberately_mismatched"),
    ):
        distilled = _target_model(
            target_backbone,
            initial_adapter_state,
            source_head_state,
            config.shared_size,
            device,
        )
        head_before = _state_copy(distilled.head)
        history = train_distillation_batched(
            source_model,
            distilled,
            split_examples["train"],
            objective_outputs,
            distillation_config,
            batch_size=config.train_batch_size,
        )
        _assert_state_unchanged(source_state["adapter"], source_model.adapter)
        _assert_state_unchanged(source_state["head"], source_model.head)
        _assert_state_unchanged(head_before, distilled.head)
        results[condition] = _record_condition(
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
            False,
            {"epoch_losses": list(history.epoch_losses)},
            {
                "head": "../../../source/head.safetensors",
                "decision_labels_used_for_adapter": "false",
                "teacher_outputs": teacher_kind,
            },
        )

    comparisons = {
        split: _comparison_tree(
            results["decision_space_distillation"]["metrics"][split],
            results["untrained_target_adapter"]["metrics"][split],
            results["mismatched_teacher_distillation"]["metrics"][split],
        )
        for split in ("in_distribution", "out_of_distribution")
    }
    return {
        "backbone_type": target.backbone_type,
        "model_id": target.model_id,
        "source_head_sha256": source_head_sha256,
        "conditions": results,
        "comparisons": comparisons,
        "matched_adapter_initialization": True,
        "adapter_initialization_sha256": initialization_sha256,
    }


def _record_condition(
    model,
    condition,
    output_dir,
    eval_examples,
    ood_examples,
    split_examples,
    teacher_outputs,
    config,
    distillation_config,
    seed,
    include_head,
    training,
    metadata=None,
):
    save_artifact(
        model,
        output_dir / condition,
        include_head=include_head,
        metadata={"role": condition, **(metadata or {})},
    )
    return {
        "training": training,
        "metrics": _evaluate_all(model, eval_examples, ood_examples, config, seed),
        "decision_match": {
            split: measure_match_stratified(
                model,
                split_examples[split],
                teacher_outputs[split],
                distillation_config,
                batch_size=config.eval_batch_size,
            )
            for split in split_examples
        },
        "artifact": condition,
    }


def aggregate_broad_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate every numeric leaf with mean and sample standard deviation."""

    if not results:
        raise ValueError("at least one result is required")
    seeds = [int(result["seed"]) for result in results]
    targets = tuple(results[0]["targets"])
    if any(tuple(result["targets"]) != targets for result in results):
        raise ValueError("all seeds must contain identical targets")
    aggregate = {
        "status": "broad_diagnostic_pending_interpretation",
        "seeds": seeds,
        "seed_count": len(seeds),
        "standard_deviation": "sample (n-1)",
        "source": {
            "metrics": _aggregate_trees([result["source"]["metrics"] for result in results])
        },
        "targets": {},
        "source_head_sha256_by_seed": {
            str(result["seed"]): result["source"]["head_sha256"] for result in results
        },
    }
    for target in targets:
        entries = [result["targets"][target] for result in results]
        aggregate["targets"][target] = {
            "model_id": entries[0]["model_id"],
            "metrics": {
                condition: _aggregate_trees(
                    [entry["conditions"][condition]["metrics"] for entry in entries]
                )
                for condition in TARGET_CONDITIONS
            },
            "decision_match": {
                condition: _aggregate_trees(
                    [entry["conditions"][condition]["decision_match"] for entry in entries]
                )
                for condition in TARGET_CONDITIONS
            },
            "comparisons": _aggregate_trees([entry["comparisons"] for entry in entries]),
        }
    return aggregate


def _evaluate_all(model, eval_examples, ood_examples, config, seed):
    return {
        "in_distribution": evaluate_stratified(
            model,
            eval_examples,
            batch_size=config.eval_batch_size,
            permutation_trials=config.permutation_trials,
            seed=seed,
        ),
        "out_of_distribution": evaluate_stratified(
            model,
            ood_examples,
            batch_size=config.eval_batch_size,
            permutation_trials=config.permutation_trials,
            seed=seed,
        ),
    }


def _comparison_tree(correct, untrained, mismatched):
    result = {}
    for key in ("overall", "by_dataset", "by_task_family", "by_decision_type"):
        if key == "overall":
            result[key] = _accuracy_gains(correct[key], untrained[key], mismatched[key])
        else:
            result[key] = {
                group: _accuracy_gains(
                    correct[key][group], untrained[key][group], mismatched[key][group]
                )
                for group in correct[key]
            }
    result["macro_across_task_families"] = _accuracy_gains(
        correct["macro_across_task_families"],
        untrained["macro_across_task_families"],
        mismatched["macro_across_task_families"],
    )
    return result


def _accuracy_gains(correct, untrained, mismatched):
    return {
        "correct_teacher_accuracy": float(correct["accuracy"]),
        "untrained_accuracy": float(untrained["accuracy"]),
        "mismatched_teacher_accuracy": float(mismatched["accuracy"]),
        "gain_over_untrained": float(correct["accuracy"] - untrained["accuracy"]),
        "gain_over_mismatched_teacher": float(correct["accuracy"] - mismatched["accuracy"]),
    }


def _aggregate_trees(values):
    first = values[0]
    if isinstance(first, dict):
        if any(not isinstance(value, dict) or value.keys() != first.keys() for value in values):
            raise ValueError("aggregate result trees differ")
        return {key: _aggregate_trees([value[key] for value in values]) for key in first}
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        numbers = [float(value) for value in values]
        return {
            "mean": statistics.mean(numbers),
            "standard_deviation": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        }
    raise TypeError(f"cannot aggregate result leaf {type(first).__name__}")


def _load_cached_backbone(backbone_type, model_id, config, device, examples):
    backbone = backbone_type.from_pretrained(
        model_id, max_length=config.max_length, model_kwargs={"dtype": "auto"}
    ).to(device)
    cached = CachedBackbone(backbone)
    additions = cached.prefill(examples, batch_size=config.cache_batch_size)
    if additions <= 0:
        raise RuntimeError("backbone cache unexpectedly remained empty")
    cached.backbone.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached


def _target_model(backbone, adapter_state, head_state, shared_size, device):
    adapter = BackboneAdapter(backbone.hidden_size, shared_size).to(device)
    adapter.load_state_dict(adapter_state)
    head = DecisionHead(shared_size).to(device)
    if head_state is not None:
        head.load_state_dict(head_state)
    return DecPort(backbone, adapter, head)


def _validate_config(config):
    if (
        min(
            config.shared_size,
            config.max_length,
            config.cache_batch_size,
            config.train_batch_size,
            config.eval_batch_size,
        )
        <= 0
    ):
        raise ValueError("sizes and batch sizes must be positive")
    if config.seeds != (0, 1, 2, 3, 4):
        raise ValueError("the broad validation requires seeds 0-4")
    if len(config.targets) != 3 or len({target.name for target in config.targets}) != 3:
        raise ValueError("the broad validation requires three unique targets")
    if config.decision_training.epochs != config.distillation_training.epochs:
        raise ValueError("all conditions must use the same epoch count")
    if (
        config.decision_training.learning_rate != config.distillation_training.learning_rate
        or config.decision_training.weight_decay != config.distillation_training.weight_decay
    ):
        raise ValueError("all conditions must use one optimizer configuration")


def _validate_dataset_contract(train, evaluation, ood):
    expected_counts = (4500, 2000, 1500)
    if tuple(map(len, (train, evaluation, ood))) != expected_counts:
        raise ValueError(f"broad split counts must be {expected_counts}")
    for split in (train, evaluation, ood):
        types = {example.decision_type for example in split}
        if types != {"choice", "boolean", "score"}:
            raise ValueError("every split must contain Choice, Boolean, and Score decisions")
        if any(not example.dataset or not example.task_family for example in split):
            raise ValueError("every broad example requires dataset and task-family metadata")


def _training_for_seed(config, seed):
    values = asdict(config)
    values["seed"] = seed
    return TrainingConfig(**values)


def _distillation_for_seed(config, seed):
    values = asdict(config)
    values["seed"] = seed
    return DistillationConfig(**values)


def _config_to_dict(config, seed):
    value = asdict(config)
    value["seed"] = seed
    return value


def _state_copy(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_state_unchanged(before, module):
    after = module.state_dict()
    if before.keys() != after.keys() or any(
        not torch.equal(before[key], after[key].detach().cpu()) for key in before
    ):
        raise RuntimeError("a frozen component changed during target training")


def _state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _progress(message):
    print(message, file=sys.stderr, flush=True)
