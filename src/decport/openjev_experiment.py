"""Open-Jev 2B DecisionCore transfer across heterogeneous frozen target backbones.

One pinned Open-Jev 2B model is the teacher. Its trained decision head and saved calibration form
exactly one frozen DecisionCore shared by every target. Each frozen target backbone reaches that
core only through a trainable DecPort adapter, optimized against Open-Jev's calibrated typed
distributions without target labels. Choice, Noul, and Score keep Open-Jev semantics and are
reported separately.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path

import torch

from decport.adapter import BackboneAdapter
from decport.backbones import CachedBackbone, GemmaBackbone, LlamaBackbone, SmolLMBackbone
from decport.broad import (
    _typed_group,
    evaluate_typed,
    predict_core_logits,
    train_core_distillation_batched,
    typed_match,
)
from decport.broad_experiment import TARGET_BACKBONES, BroadTargetSpec, _aggregate_trees
from decport.data import load_jsonl
from decport.decision_core import (
    DECISION_KINDS,
    LinearDecisionCore,
    TypedDecision,
    typed_from_example,
)
from decport.distillation import DistillationConfig, TeacherOutput, permute_teacher_outputs
from decport.experiment import resolve_device
from decport.model import DecPort
from decport.openjev import (
    OPENJEV_2B_MANIFEST_SHA256,
    OPENJEV_2B_REPO_ID,
    OPENJEV_2B_REVISION,
    OPENJEV_COMMIT,
    OPENJEV_REPOSITORY,
    OpenJevTeacher,
    download_openjev_package,
    installed_openjev_commit,
    load_openjev_core,
    render_openjev_prompts,
    verify_openjev_package,
)
from decport.serialization import save_artifact

CONDITIONS = (
    "untrained_target_adapter",
    "openjev_teacher_distillation",
    "mismatched_teacher_distillation",
    "random_core_distillation",
)
CONTROLS = {
    "untrained_target_adapter": "untrained",
    "mismatched_teacher_distillation": "mismatched_teacher",
    "random_core_distillation": "random_core",
}
SPLITS = ("train", "in_distribution", "out_of_distribution")
EVALUATION_SPLITS = SPLITS[1:]
FULL_SPLIT_COUNTS = {"train": 4500, "in_distribution": 2000, "out_of_distribution": 1500}
REVIEW_LEVELS = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")


@dataclass(frozen=True, slots=True)
class OpenJevTransferConfig:
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
    package_dir: str | None = None
    package_manifest_sha256: str = OPENJEV_2B_MANIFEST_SHA256
    adapter_hidden_size: int = 256
    teacher_max_length: int = 4096
    teacher_candidate_batch_size: int = 8
    target_max_length: int = 2048
    cache_batch_size: int = 16
    train_batch_size: int = 8
    eval_batch_size: int = 64
    device: str = "auto"
    distillation: DistillationConfig = field(
        default_factory=lambda: DistillationConfig(epochs=10, temperature=1.0)
    )
    # Bounded smoke slices only: the first N decisions of each type per split.
    limit_per_type: dict[str, int] | None = None

    @property
    def is_smoke(self) -> bool:
        return self.limit_per_type is not None


def run_openjev_experiment(config: OpenJevTransferConfig) -> dict[str, object]:
    """Run the teacher pass, cache targets, run every seed/target/condition, and aggregate."""

    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    paths = {
        "train": config.train_path,
        "in_distribution": config.eval_path,
        "out_of_distribution": config.ood_path,
    }
    labeled = {
        split: _load_typed(path, config.limit_per_type, split) for split, path in paths.items()
    }
    _validate_data(labeled, bounded=config.is_smoke)
    unlabeled = {split: [item.without_answer() for item in labeled[split]] for split in SPLITS}
    telemetry: dict[str, object] = {"device": str(device), "targets": {}}

    installed_commit = installed_openjev_commit()
    if installed_commit != OPENJEV_COMMIT:
        raise RuntimeError(f"installed Open-Jev is {installed_commit}, expected {OPENJEV_COMMIT}")
    package = verify_openjev_package(
        Path(config.package_dir) if config.package_dir else download_openjev_package(),
        expected_manifest_sha256=config.package_manifest_sha256,
    )
    core = load_openjev_core(package).to(device)
    core_sha256 = core.state_sha256()

    _progress("rendering Open-Jev candidate prompts")
    prompts = {
        id(item): render_openjev_prompts(item)
        for split in SPLITS
        for item in (*labeled[split], *unlabeled[split])
    }

    def render(decision: TypedDecision) -> list[str]:
        return prompts[id(decision)]

    _progress("running the frozen Open-Jev 2B teacher once for all seeds and targets")
    raw_logits, telemetry["teacher"] = _teacher_pass(config, package, core, unlabeled, device)
    teacher_outputs = {
        split: tuple(TeacherOutput(scores=row / core.temperature) for row in raw_logits[split])
        for split in SPLITS
    }
    teacher = {
        "metrics": {
            split: evaluate_typed(
                labeled[split], [output.scores.tolist() for output in teacher_outputs[split]]
            )
            for split in EVALUATION_SPLITS
        },
        "distribution": {
            split: _distribution_summary(unlabeled[split], teacher_outputs[split])
            for split in SPLITS
        },
    }
    _write_json(output_dir / "teacher" / "summary.json", teacher)
    _write_json(
        output_dir / "teacher" / "raw_logits.json",
        {split: [row.tolist() for row in raw_logits[split]] for split in SPLITS},
    )

    all_prompts = {prompt for rendered in prompts.values() for prompt in rendered}
    target_backbones = {}
    target_revisions = {}
    for target in config.targets:
        _progress(f"caching frozen {target.name} representations of Open-Jev prompts")
        backbone, target_revisions[target.name], telemetry["targets"][target.name] = (
            _cache_target(config, target, all_prompts, device)
        )
        target_backbones[target.name] = backbone

    results = []
    seed_seconds = {}
    _reset_peak(device)
    for seed in config.seeds:
        _progress(f"starting seed {seed}")
        started = time.perf_counter()
        results.append(
            _run_seed(
                config,
                seed,
                output_dir / f"seed-{seed}",
                core,
                core_sha256,
                target_backbones,
                labeled,
                unlabeled,
                teacher_outputs,
                render,
                device,
            )
        )
        seed_seconds[str(seed)] = time.perf_counter() - started
    telemetry["seed_seconds"] = seed_seconds
    telemetry["peak_gpu_memory_mib_training"] = _peak_mib(device)

    summary = aggregate_openjev_results(results)
    summary["teacher"] = teacher
    _write_json(output_dir / "aggregate_summary.json", summary)
    _write_json(output_dir / "telemetry.json", telemetry)
    _write_json(
        output_dir / "provenance.json",
        _provenance(config, package, core, core_sha256, installed_commit, target_revisions),
    )
    return summary


def aggregate_openjev_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate every numeric leaf with mean and sample standard deviation."""

    if not results:
        raise ValueError("at least one result is required")
    targets = tuple(results[0]["targets"])
    if any(tuple(result["targets"]) != targets for result in results):
        raise ValueError("all seeds must contain identical targets")
    core_hashes = {result["decision_core"]["sha256"] for result in results}
    if len(core_hashes) != 1:
        raise ValueError("every seed must use the same frozen decision core")
    aggregate: dict[str, object] = {
        "status": results[0]["status"],
        "seeds": [result["seed"] for result in results],
        "seed_count": len(results),
        "standard_deviation": "sample (n-1)",
        "decision_core_sha256": core_hashes.pop(),
        "targets": {},
    }
    for target in targets:
        entries = [result["targets"][target] for result in results]
        aggregate["targets"][target] = {
            "model_id": entries[0]["model_id"],
            "metrics": {
                condition: _aggregate_trees(
                    [entry["conditions"][condition]["metrics"] for entry in entries]
                )
                for condition in CONDITIONS
            },
            "decision_match": {
                condition: _aggregate_trees(
                    [entry["conditions"][condition]["decision_match"] for entry in entries]
                )
                for condition in CONDITIONS
            },
            "comparisons": _aggregate_trees([entry["comparisons"] for entry in entries]),
        }
    return aggregate


def _teacher_pass(config, package, core, unlabeled, device):
    _reset_peak(device)
    started = time.perf_counter()
    teacher_device = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
    teacher = OpenJevTeacher.from_package(
        package, core, device=teacher_device, max_length=config.teacher_max_length
    )
    raw = {
        split: teacher.raw_logits(
            unlabeled[split], candidate_batch_size=config.teacher_candidate_batch_size
        )
        for split in SPLITS
    }
    telemetry = {
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": _peak_mib(device),
        "candidate_sequences": teacher.candidate_sequences,
        "input_tokens": teacher.input_tokens,
        "core_parity_max_abs_diff": teacher.parity_max_abs_diff,
        "max_length": config.teacher_max_length,
        "candidate_batch_size": config.teacher_candidate_batch_size,
    }
    del teacher
    _release(device)
    return raw, telemetry


def _cache_target(config, target, prompts, device):
    _reset_peak(device)
    started = time.perf_counter()
    backbone = TARGET_BACKBONES[target.backbone_type].from_pretrained(
        target.model_id,
        max_length=config.target_max_length,
        allow_truncation=False,
        model_kwargs={"dtype": "auto"},
    ).to(device)
    revision = getattr(backbone.model.config, "_commit_hash", None)
    lengths = [len(ids) for ids in backbone.tokenizer(sorted(prompts))["input_ids"]]
    cached = CachedBackbone(backbone)
    if cached.prefill_prompts(prompts, batch_size=config.cache_batch_size) != len(prompts):
        raise RuntimeError("target cache did not add every rendered prompt")
    cached.backbone.to("cpu")
    telemetry = {
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": _peak_mib(device),
        "prompts": len(prompts),
        "max_prompt_tokens": max(lengths),
        "mean_prompt_tokens": sum(lengths) / len(lengths),
        "hidden_size": cached.hidden_size,
        "dtype": str(next(backbone.model.parameters()).dtype),
    }
    _release(device)
    return cached, revision, telemetry


def _run_seed(
    config,
    seed,
    output_dir,
    core,
    core_sha256,
    target_backbones,
    labeled,
    unlabeled,
    teacher_outputs,
    render,
    device,
):
    distillation = replace(config.distillation, seed=seed)
    random_core = LinearDecisionCore.random_control(core, seed=_derived_seed(seed, "random-core"))
    random_core = random_core.to(device)
    mismatched = permute_teacher_outputs(
        unlabeled["train"], teacher_outputs["train"], seed=seed, group_key=_typed_group
    )
    targets = {}
    for target in config.targets:
        _progress(f"seed {seed}: target {target.name}")
        targets[target.name] = _run_target(
            config,
            target,
            seed,
            output_dir / "targets" / target.name,
            core,
            random_core,
            target_backbones[target.name],
            labeled,
            unlabeled,
            teacher_outputs,
            mismatched,
            distillation,
            render,
            device,
        )
    core.assert_frozen()
    if core.state_sha256() != core_sha256:
        raise RuntimeError("the Open-Jev decision core changed")
    result = {
        "status": _status(config),
        "seed": seed,
        "config": _config_to_dict(config, seed),
        "decision_core": {
            "source": f"{OPENJEV_2B_REPO_ID}@{OPENJEV_2B_REVISION}",
            "sha256": core_sha256,
            "input_size": core.input_size,
            "temperature": core.temperature,
            "shared_by_targets": [target.name for target in config.targets],
        },
        "random_control_core_sha256": random_core.state_sha256(),
        "targets": targets,
        "label_free_audit": {
            "transfer_entry_point_rejects_labeled_records": True,
            "distillation_decisions": len(unlabeled["train"]),
            "distillation_decisions_with_answers": sum(
                item.answer is not None for item in unlabeled["train"]
            ),
            "teacher_inputs_with_answers": sum(
                item.answer is not None for split in SPLITS for item in unlabeled[split]
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
                for condition in CONDITIONS[1:]
            ],
            "frozen_components": [
                "openjev_qwen3.5_2b_backbone",
                "openjev_lora_adapter",
                "openjev_decision_head",
                "openjev_calibration_temperature",
                "random_control_core",
                *[f"{target.name}_backbone" for target in config.targets],
            ],
            "same_openjev_core_sha256_for_all_targets": core_sha256,
            "openjev_core_verified_unchanged": True,
        },
    }
    _write_json(output_dir / "results.json", result)
    return result


def _run_target(
    config,
    target,
    seed,
    output_dir,
    core,
    random_core,
    backbone,
    labeled,
    unlabeled,
    teacher_outputs,
    mismatched,
    distillation,
    render,
    device,
):
    torch.manual_seed(_derived_seed(seed, target.name))
    initial = BackboneAdapter(
        backbone.hidden_size, core.input_size, hidden_size=config.adapter_hidden_size
    )
    initial_state = _state_copy(initial)
    conditions = {}

    untrained = _target_model(backbone, initial_state, core, config, device)
    _assert_guard_rejects_labels(untrained, labeled["train"], teacher_outputs["train"], render)
    conditions["untrained_target_adapter"] = _record(
        untrained, "untrained_target_adapter", output_dir, labeled, unlabeled, teacher_outputs,
        config, distillation, render, None, core=core,
    )
    for condition, condition_core, objective, teacher_kind in (
        ("openjev_teacher_distillation", core, teacher_outputs["train"], "matched"),
        ("mismatched_teacher_distillation", core, mismatched, "deranged_within_type_and_width"),
        ("random_core_distillation", random_core, teacher_outputs["train"], "matched"),
    ):
        model = _target_model(backbone, initial_state, condition_core, config, device)
        started = time.perf_counter()
        history = train_core_distillation_batched(
            model,
            unlabeled["train"],
            objective,
            distillation,
            render=render,
            batch_size=config.train_batch_size,
        )
        training = {
            "teacher_outputs": teacher_kind,
            "seconds": time.perf_counter() - started,
            "epoch_losses": list(history.epoch_losses),
            "epoch_kl_divergence": list(history.epoch_kl_divergence),
            "epoch_centered_logit_mse": list(history.epoch_centered_logit_mse),
            "gradient_audit": history.gradient_audit,
        }
        conditions[condition] = _record(
            model, condition, output_dir, labeled, unlabeled, teacher_outputs, config,
            distillation, render, training, core=condition_core,
        )
    return {
        "backbone_type": target.backbone_type,
        "model_id": target.model_id,
        "adapter": {
            "input_size": initial.input_size,
            "hidden_size": initial.hidden_size,
            "output_size": initial.shared_size,
            "parameter_count": initial.parameter_count,
            "matched_initialization_sha256": _state_sha256(initial_state),
        },
        "conditions": conditions,
        "comparisons": {split: _comparisons(conditions, split) for split in EVALUATION_SPLITS},
    }


def _record(
    model,
    condition,
    output_dir,
    labeled,
    unlabeled,
    teacher_outputs,
    config,
    distillation,
    render,
    training,
    *,
    core,
):
    random_core = condition == "random_core_distillation"
    save_artifact(
        model,
        output_dir / condition,
        include_head=random_core,
        metadata={
            "role": condition,
            "decision_core": "random_control_core" if random_core else "openjev_2b_head",
            "decision_core_sha256": core.state_sha256(),
            "decision_labels_used_for_adapter": "false",
            "head_format": "decport.LinearDecisionCore" if random_core else "not_included",
        },
    )
    logits = {
        split: predict_core_logits(
            model, unlabeled[split], render=render, batch_size=config.eval_batch_size
        )
        for split in SPLITS
    }
    return {
        "training": training,
        "metrics": {
            split: evaluate_typed(labeled[split], logits[split]) for split in EVALUATION_SPLITS
        },
        "decision_match": {
            split: typed_match(
                unlabeled[split], logits[split], teacher_outputs[split], distillation
            )
            for split in SPLITS
        },
        "artifact": condition,
    }


def _comparisons(conditions, split):
    """Accuracy and teacher-agreement gains of the correct teacher over each control."""

    def gains(select):
        correct = select(conditions["openjev_teacher_distillation"])
        values = {
            "correct_teacher_accuracy": float(correct[0]),
            "correct_teacher_agreement": float(correct[1]),
        }
        for condition, name in CONTROLS.items():
            control = select(conditions[condition])
            values[f"{name}_accuracy"] = float(control[0])
            values[f"{name}_agreement"] = float(control[1])
            values[f"accuracy_gain_over_{name}"] = float(correct[0] - control[0])
            values[f"agreement_gain_over_{name}"] = float(correct[1] - control[1])
        return values

    def pick(group, key=None):
        def select(condition):
            metrics = condition["metrics"][split][group]
            match = condition["decision_match"][split][group]
            if key is not None:
                metrics, match = metrics[key], match[key]
            return metrics["accuracy"], match["top_choice_agreement"]

        return select

    reference = conditions["openjev_teacher_distillation"]["metrics"][split]
    return {
        "overall": gains(pick("overall")),
        "macro_across_decision_types": gains(pick("macro_across_decision_types")),
        "by_decision_type": {
            kind: gains(pick("by_decision_type", kind)) for kind in reference["by_decision_type"]
        },
        "by_dataset": {name: gains(pick("by_dataset", name)) for name in reference["by_dataset"]},
    }


def _distribution_summary(decisions, outputs):
    """Per-type evidence that teacher distributions are real and not degenerate."""

    summary = {}
    for kind in DECISION_KINDS:
        rows = [
            torch.softmax(output.scores.double(), dim=0)
            for decision, output in zip(decisions, outputs, strict=True)
            if decision.kind == kind
        ]
        if not rows:
            continue
        maxima = [float(row.max()) for row in rows]
        entropies = [
            float(-(row * row.clamp_min(1e-300).log()).sum()) / math.log(len(row)) for row in rows
        ]
        argmax = [int(row.argmax()) for row in rows]
        distinct = len({tuple(round(float(value), 6) for value in row) for row in rows})
        if distinct < 2 or not all(math.isfinite(value) for value in maxima):
            raise RuntimeError(f"Open-Jev {kind} teacher distributions are degenerate")
        count = len(rows)
        values = {
            "count": count,
            "distinct_distributions": distinct,
            "mean_max_probability": sum(maxima) / count,
            "mean_normalized_entropy": sum(entropies) / count,
            "fraction_max_probability_above_0_99": sum(value > 0.99 for value in maxima) / count,
            "argmax_index_counts": {
                str(index): argmax.count(index) for index in sorted(set(argmax))
            },
        }
        if kind == "noul":
            values["mean_p_true"] = sum(float(row[1]) for row in rows) / count
            values["fraction_p_true_above_0_5"] = sum(float(row[1]) > 0.5 for row in rows) / count
        summary[kind] = values
    return summary


def _assert_guard_rejects_labels(model, labeled_train, teacher_train, render):
    """Exercise the hard label guard on the real transfer entry point before any training."""

    try:
        train_core_distillation_batched(
            model,
            [labeled_train[0]],
            (teacher_train[0],),
            DistillationConfig(epochs=1),
            render=render,
            batch_size=1,
        )
    except ValueError as error:
        if "must not include answers" in str(error):
            return
        raise
    raise RuntimeError("the transfer entry point accepted a labeled record")


def _load_typed(path, limit_per_type, split):
    decisions = [typed_from_example(example) for example in load_jsonl(path)]
    if limit_per_type is None:
        return decisions
    limit = limit_per_type[split]
    selected, counts = [], dict.fromkeys(DECISION_KINDS, 0)
    for decision in decisions:
        if counts[decision.kind] < limit:
            counts[decision.kind] += 1
            selected.append(decision)
    return selected


def _validate_config(config):
    sizes = (
        config.adapter_hidden_size,
        config.teacher_max_length,
        config.teacher_candidate_batch_size,
        config.target_max_length,
        config.cache_batch_size,
        config.train_batch_size,
        config.eval_batch_size,
    )
    if min(sizes) <= 0:
        raise ValueError("sizes, lengths, and batch sizes must be positive")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    names = [target.name for target in config.targets]
    if not names or len(set(names)) != len(names):
        raise ValueError("targets must be non-empty and uniquely named")
    if config.limit_per_type is not None and (
        set(config.limit_per_type) != set(SPLITS)
        or min(config.limit_per_type.values()) < 2
    ):
        raise ValueError("limit_per_type needs every split and at least two decisions per type")


def _validate_data(labeled, *, bounded):
    for split, decisions in labeled.items():
        if {decision.kind for decision in decisions} != set(DECISION_KINDS):
            raise ValueError(f"{split} must contain Choice, Noul, and Score decisions")
        if any(decision.answer is None or not decision.dataset for decision in decisions):
            raise ValueError(f"{split} needs labels and dataset metadata for evaluation")
        if any(
            decision.kind == "score" and decision.criteria != REVIEW_LEVELS
            for decision in decisions
        ):
            raise ValueError("Score levels must be the ordered five-level review scale")
        if not bounded and len(decisions) != FULL_SPLIT_COUNTS[split]:
            raise ValueError(f"{split} must contain {FULL_SPLIT_COUNTS[split]} decisions")


def _target_model(backbone, adapter_state, core, config, device):
    adapter = BackboneAdapter(
        backbone.hidden_size, core.input_size, hidden_size=config.adapter_hidden_size
    ).to(device)
    adapter.load_state_dict(adapter_state)
    return DecPort(backbone, adapter, core)


def _provenance(config, package, core, core_sha256, installed_commit, target_revisions):
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    repository = Path(__file__).resolve().parents[2]
    return {
        "decport": {
            "git_commit": _git(repository, "rev-parse", "HEAD"),
            "git_worktree_dirty": bool(_git(repository, "status", "--porcelain")),
        },
        "openjev": {
            "repository": OPENJEV_REPOSITORY,
            "pinned_commit": OPENJEV_COMMIT,
            "installed_commit": installed_commit,
            "model_repository": OPENJEV_2B_REPO_ID,
            "model_revision": OPENJEV_2B_REVISION,
            "package": package.provenance(),
            "decision_core_sha256": core_sha256,
            "decision_core_input_size": core.input_size,
            "calibration_temperature": core.temperature,
        },
        "teacher_base_model": {"model_id": package.model_id, "revision": package.revision},
        "targets": {
            target.name: {
                "model_id": target.model_id,
                "resolved_revision": target_revisions[target.name],
            }
            for target in config.targets
        },
        "data": {
            split: {"path": path, "sha256": _file_sha256(path)}
            for split, path in (
                ("train", config.train_path),
                ("in_distribution", config.eval_path),
                ("out_of_distribution", config.ood_path),
            )
        },
        "packages": {
            name: version(name)
            for name in (
                "torch", "transformers", "peft", "accelerate", "safetensors",
                "huggingface-hub", "tokenizers", "datasets", "open-jev",
            )
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _status(config):
    if config.is_smoke:
        return "smoke_test_not_evidence"
    return "openjev_transfer_pending_interpretation"


def _config_to_dict(config, seed):
    value = asdict(config)
    value["seed"] = seed
    return value


def _derived_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _state_copy(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mib(device):
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


def _release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _progress(message):
    print(message, file=sys.stderr, flush=True)
