"""DecPort final ship gate (decision 0007).

One compact nonlinear DecisionCore is learned once, with labels, on source-only records over frozen
Open-Jev 2B representations, then frozen permanently. Each heterogeneous frozen target reaches it
only through a deliberately weak rank-128 linear adapter trained label-free against the source
system's calibrated typed outputs. A random core of the same architecture and scale, a mismatched
teacher, an untrained adapter, and a target-specific distilled module are the controls. The
pre-registered SHIP criteria are implemented by :func:`evaluate_ship_gate`.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from decport.adapter import LowRankAdapter
from decport.backbones import GemmaBackbone, LlamaBackbone, SmolLMBackbone
from decport.broad import (
    _indexed_batches_by_option_count,
    _typed_group,
    evaluate_typed,
    predict_core_logits,
    train_core_distillation_batched,
    typed_match,
)
from decport.broad_experiment import BroadTargetSpec, _aggregate_trees
from decport.decision_core import (
    DECISION_KINDS,
    FrozenDecisionCore,
    MLPDecisionCore,
    ScalarIdentityCore,
    TypedDecision,
    mlp_decision_network,
    typed_logits,
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
from decport.openjev_experiment import (
    REVIEW_LEVELS,
    _cache_target,
    _derived_seed,
    _distribution_summary,
    _file_sha256,
    _git,
    _load_typed,
    _peak_mib,
    _release,
    _reset_peak,
    _state_copy,
    _state_sha256,
    _write_json,
)

CONDITIONS = (
    "untrained_adapter",
    "learned_core_distillation",
    "mismatched_teacher_distillation",
    "random_core_distillation",
    "target_specific_distillation",
)
ADAPTER_CONDITIONS = CONDITIONS[:4]
TRAINED_CONDITIONS = CONDITIONS[1:]
CONTROLS = {
    "random_core_distillation": "random_core",
    "mismatched_teacher_distillation": "mismatched_teacher",
    "untrained_adapter": "untrained",
    "target_specific_distillation": "target_specific",
}
SOURCE_SPLITS = ("source_core_train", "source_core_calibration")
TRANSFER_SPLITS = ("train", "in_distribution", "out_of_distribution")
EVALUATION_SPLITS = TRANSFER_SPLITS[1:]
ALL_SPLITS = SOURCE_SPLITS + TRANSFER_SPLITS
FULL_SPLIT_COUNTS = {
    "source_core_train": 4500,
    "source_core_calibration": 900,
    "train": 4500,
    "in_distribution": 2000,
    "out_of_distribution": 1500,
}
RANDOM_CORE_RULE = (
    "same architecture and temperature; Gaussian tensors rescaled to the learned tensors' "
    "Frobenius norms, then one gain and one shift per Linear layer so its pre-activation mean and "
    "std on the source-core training representations equal the learned core's (layer by layer)"
)
STATUS_FULL = "final_gate_pending_verdict"
STATUS_SMOKE = "smoke_test_not_evidence"

# Pre-registered thresholds (decision 0007). Never change after the full run starts.
SHIP_THRESHOLDS = {
    "min_targets": 2,
    "min_positive_seeds": 4,
    "core_gain_id": 0.05,
    "core_gain_ood": 0.03,
    "clearly_negative_core_gain_id": -0.02,
    "mismatch_gain_id": 0.05,
    "ood_transfer_gain": 0.03,
    "jevbench_margin_over_uniform": 0.05,
    "baseline_match_margin": 0.02,
    "substantially_fewer_parameter_ratio": 0.5,
}


@dataclass(frozen=True, slots=True)
class SourceCoreConfig:
    seed: int = 0
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    temperature_grid: tuple[float, float, int] = (0.05, 20.0, 4001)

    def __post_init__(self) -> None:
        low, high, points = self.temperature_grid
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("source-core epochs, batch size, and learning rate must be positive")
        if not 0 < low < high or points < 2:
            raise ValueError("the temperature grid must be increasing, positive, and non-trivial")


@dataclass(frozen=True, slots=True)
class FinalGateConfig:
    train_path: str
    eval_path: str
    ood_path: str
    source_train_path: str
    source_calibration_path: str
    output_dir: str
    targets: tuple[BroadTargetSpec, ...] = (
        BroadTargetSpec("smollm", "smollm", SmolLMBackbone.DEFAULT_MODEL_ID),
        BroadTargetSpec("gemma", "gemma", GemmaBackbone.DEFAULT_MODEL_ID),
        BroadTargetSpec("tinyllama", "llama", LlamaBackbone.DEFAULT_MODEL_ID),
    )
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    package_dir: str | None = None
    package_manifest_sha256: str = OPENJEV_2B_MANIFEST_SHA256
    adapter_rank: int = 128
    core_hidden_sizes: tuple[int, int] = (512, 128)
    baseline_hidden_sizes: tuple[int, int] = (512, 128)
    teacher_max_length: int = 4096
    teacher_candidate_batch_size: int = 8
    target_max_length: int = 2048
    cache_batch_size: int = 16
    train_batch_size: int = 8
    eval_batch_size: int = 64
    device: str = "auto"
    source_core: SourceCoreConfig = field(default_factory=SourceCoreConfig)
    distillation: DistillationConfig = field(
        default_factory=lambda: DistillationConfig(epochs=10, temperature=1.0)
    )
    # Bounded smoke slices only: the first N decisions of each type per split.
    limit_per_type: dict[str, int] | None = None

    @property
    def is_smoke(self) -> bool:
        return self.limit_per_type is not None


class TargetDecisionModule(nn.Module):
    """Target-specific baseline: the core architecture trained directly on one backbone's states.

    It emits one scalar per candidate and is scored through a :class:`ScalarIdentityCore`, so it
    uses the same typed assembly, calibration constant, objective, and trainer as DecPort, but does
    not reuse the shared core.
    """

    def __init__(self, input_size: int, hidden_sizes: tuple[int, int] = (512, 128)) -> None:
        super().__init__()
        self.input_size = input_size
        self.shared_size = 1
        self.hidden_sizes = tuple(hidden_sizes)
        self.network = mlp_decision_network(input_size, self.hidden_sizes)

    @property
    def hidden_size(self) -> int:
        return self.hidden_sizes[0]

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.input_size:
            raise ValueError(f"expected hidden width {self.input_size}, got {hidden.shape[-1]}")
        parameter = next(self.parameters())
        return self.network(hidden.to(device=parameter.device, dtype=parameter.dtype))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True, slots=True)
class SourceCoreTraining:
    state: dict[str, Tensor]
    epoch_losses: tuple[float, ...]
    epoch_accuracy: tuple[float, ...]
    seconds: float


def train_source_core(
    decisions: list[TypedDecision],
    representations: list[Tensor],
    config: SourceCoreConfig,
    *,
    input_size: int,
    hidden_sizes: tuple[int, int],
    device: torch.device,
) -> SourceCoreTraining:
    """Supervised source-side training of the nonlinear core on labeled source-only records.

    Typed cross-entropy on uncalibrated typed logits (Noul ``[0, s]`` is binary cross-entropy),
    fixed epochs, no early stopping or selection.
    """

    if not decisions or any(decision.answer is None for decision in decisions):
        raise ValueError("source-core training requires labeled source decisions")
    if len(decisions) != len(representations):
        raise ValueError("one representation block is required per source decision")
    for decision, hidden in zip(decisions, representations, strict=True):
        if tuple(hidden.shape) != (decision.candidate_count, input_size):
            raise ValueError("source representations must be (candidates, input_size)")
    started = time.perf_counter()
    torch.manual_seed(_derived_seed(config.seed, "source-core"))
    network = mlp_decision_network(input_size, hidden_sizes).to(device)
    network.train()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = random.Random(config.seed)
    indexed = list(enumerate(decisions))
    losses, accuracies = [], []
    for _ in range(config.epochs):
        batches = _indexed_batches_by_option_count(
            indexed, config.batch_size, rng, key=_typed_group
        )
        total = 0.0
        correct = count = 0
        for batch in batches:
            kind, width = batch[0][1].kind, batch[0][1].candidate_count
            hidden = torch.cat([representations[index] for index, _ in batch]).to(device)
            scores = network(hidden).squeeze(-1).reshape(len(batch), width)
            logits = typed_logits(kind, scores)
            target = torch.tensor([decision.answer_index for _, decision in batch], device=device)
            loss = nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            correct += int((logits.detach().argmax(dim=1) == target).sum())
            count += len(batch)
        losses.append(total / count)
        accuracies.append(correct / count)
    network.eval()
    state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
    return SourceCoreTraining(
        state, tuple(losses), tuple(accuracies), time.perf_counter() - started
    )


def core_calibrated_logits(
    core: FrozenDecisionCore,
    decisions: list[TypedDecision],
    representations: list[Tensor],
    *,
    batch_size: int = 256,
) -> list[Tensor]:
    """Calibrated typed logits of a frozen core on captured source representations, input order."""

    groups: dict[tuple[str, int], list[int]] = {}
    for index, decision in enumerate(decisions):
        groups.setdefault(_typed_group(decision), []).append(index)
    rows: list[Tensor | None] = [None] * len(decisions)
    device = next(iter(core.parameters()), torch.empty(0)).device
    with torch.inference_mode():
        for (kind, _), indices in groups.items():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                width = decisions[chunk[0]].candidate_count
                hidden = torch.cat([representations[index] for index in chunk]).to(device)
                scores = core(hidden).reshape(len(chunk), width)
                for index, row in zip(chunk, core.calibrated_logits(kind, scores), strict=True):
                    rows[index] = row.float().cpu().clone()
    if any(row is None for row in rows):
        raise RuntimeError("core scoring did not cover every decision")
    return [row for row in rows if row is not None]


def fit_temperature(
    decisions: list[TypedDecision],
    logits: list[Tensor],
    grid: tuple[float, float, int],
) -> dict[str, float]:
    """The single temperature minimizing mean NLL over a fixed log-spaced grid (lowest on ties)."""

    if not decisions or any(decision.answer is None for decision in decisions):
        raise ValueError("calibration requires labeled source decisions")
    low, high, points = grid
    temperatures = torch.exp(
        torch.linspace(math.log(low), math.log(high), points, dtype=torch.float64)
    )
    total = torch.zeros(points, dtype=torch.float64)
    groups: dict[tuple[str, int], list[int]] = {}
    for index, decision in enumerate(decisions):
        groups.setdefault(_typed_group(decision), []).append(index)
    for indices in groups.values():
        rows = torch.stack([logits[index].double() for index in indices])
        target = torch.tensor([decisions[index].answer_index for index in indices])
        scaled = rows.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_probability = torch.log_softmax(scaled, dim=2)
        picked = log_probability.gather(
            2, target.view(1, -1, 1).expand(points, -1, 1)
        ).squeeze(-1)
        total -= picked.sum(dim=1)
    nll = total / len(decisions)
    best = int(torch.argmin(nll))
    unit = sum(
        float(-torch.log_softmax(row.double(), dim=0)[decision.answer_index])
        for decision, row in zip(decisions, logits, strict=True)
    ) / len(decisions)
    return {
        "temperature": float(temperatures[best]),
        "calibration_nll": float(nll[best]),
        "calibration_nll_at_temperature_1": unit,
        "grid_min": low,
        "grid_max": high,
        "grid_points": points,
    }


def run_final_gate(config: FinalGateConfig) -> dict[str, object]:
    """Source pass, core training, target caching, every seed/target/condition, aggregate."""

    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    telemetry: dict[str, object] = {"device": str(device), "targets": {}}

    paths = {
        "source_core_train": config.source_train_path,
        "source_core_calibration": config.source_calibration_path,
        "train": config.train_path,
        "in_distribution": config.eval_path,
        "out_of_distribution": config.ood_path,
    }
    loaded = {
        split: _load_typed(path, config.limit_per_type, split) for split, path in paths.items()
    }
    _validate_data(loaded, bounded=config.is_smoke)
    overlap = _content_overlap(
        [*loaded["source_core_train"], *loaded["source_core_calibration"]],
        [*loaded["train"], *loaded["in_distribution"], *loaded["out_of_distribution"]],
    )
    if overlap:
        raise ValueError(f"source-core records overlap the transfer splits ({overlap})")
    # Transfer-train answers are stripped here and never used; only source and evaluation splits
    # keep labels, for source-core training and for evaluation respectively.
    transfer_train_answers_stripped = sum(item.answer is not None for item in loaded["train"])
    labeled = {split: loaded[split] for split in ALL_SPLITS if split != "train"}
    unlabeled = {split: [item.without_answer() for item in loaded[split]] for split in ALL_SPLITS}
    del loaded

    installed_commit = installed_openjev_commit()
    if installed_commit != OPENJEV_COMMIT:
        raise RuntimeError(f"installed Open-Jev is {installed_commit}, expected {OPENJEV_COMMIT}")
    package = verify_openjev_package(
        Path(config.package_dir) if config.package_dir else download_openjev_package(),
        expected_manifest_sha256=config.package_manifest_sha256,
    )
    openjev_core = load_openjev_core(package).to(device)

    _progress("running the frozen Open-Jev 2B representation pass once for every split")
    openjev_raw, representations, telemetry["teacher"] = _representation_pass(
        config, package, openjev_core, unlabeled, device
    )
    input_size = openjev_core.input_size

    _progress("training the nonlinear DecisionCore once on labeled source-only records")
    training = train_source_core(
        labeled["source_core_train"],
        representations["source_core_train"],
        config.source_core,
        input_size=input_size,
        hidden_sizes=config.core_hidden_sizes,
        device=device,
    )
    uncalibrated = MLPDecisionCore(
        input_size, 1.0, state=training.state, hidden_sizes=config.core_hidden_sizes
    ).to(device)
    calibration = fit_temperature(
        labeled["source_core_calibration"],
        core_calibrated_logits(
            uncalibrated,
            unlabeled["source_core_calibration"],
            representations["source_core_calibration"],
        ),
        config.source_core.temperature_grid,
    )
    del uncalibrated
    core = MLPDecisionCore(
        input_size,
        calibration["temperature"],
        state=training.state,
        hidden_sizes=config.core_hidden_sizes,
    ).to(device)
    core.assert_frozen()
    core_sha256 = core.state_sha256()
    core_artifact = _save_core(
        core,
        output_dir / "source_core",
        role="learned_source_core",
        extra={
            "training": {
                "epoch_losses": list(training.epoch_losses),
                "epoch_train_accuracy": list(training.epoch_accuracy),
                "seconds": training.seconds,
                "labeled_source_decisions": len(labeled["source_core_train"]),
                "config": asdict(config.source_core),
            },
            "calibration": calibration,
        },
    )

    source_logits = {
        split: core_calibrated_logits(core, unlabeled[split], representations[split])
        for split in ALL_SPLITS
    }
    teacher_outputs = {
        split: tuple(TeacherOutput(scores=row) for row in source_logits[split])
        for split in TRANSFER_SPLITS
    }
    source_system = {
        "definition": "frozen Open-Jev 2B representation -> learned DecisionCore -> calibration",
        "decision_core_sha256": core_sha256,
        "decision_core_parameters": core.parameter_count,
        "temperature": core.temperature,
        "metrics": {
            split: evaluate_typed(labeled[split], [row.tolist() for row in source_logits[split]])
            for split in (*SOURCE_SPLITS, *EVALUATION_SPLITS)
        },
        "openjev_head_reference_metrics": {
            split: evaluate_typed(
                labeled[split],
                [(row / openjev_core.temperature).tolist() for row in openjev_raw[split]],
            )
            for split in EVALUATION_SPLITS
        },
        "distribution": {
            split: _distribution_summary(unlabeled[split], teacher_outputs[split])
            for split in TRANSFER_SPLITS
        },
        "core_artifact": core_artifact,
    }
    _write_json(output_dir / "source_system" / "summary.json", source_system)
    _write_json(
        output_dir / "source_system" / "teacher_logits.json",
        {split: [row.tolist() for row in source_logits[split]] for split in TRANSFER_SPLITS},
    )
    _write_json(
        output_dir / "source_system" / "openjev_raw_logits.json",
        {split: [row.tolist() for row in openjev_raw[split]] for split in ALL_SPLITS},
    )

    prompts = {
        id(item): render_openjev_prompts(item)
        for split in TRANSFER_SPLITS
        for item in unlabeled[split]
    }

    def render(decision: TypedDecision) -> list[str]:
        cached = prompts.get(id(decision))
        return cached if cached is not None else render_openjev_prompts(decision)

    all_prompts = {prompt for rendered in prompts.values() for prompt in rendered}
    target_backbones, target_revisions, backbone_digests = {}, {}, {}
    for target in config.targets:
        _progress(f"caching frozen {target.name} representations of Open-Jev prompts")
        backbone, target_revisions[target.name], telemetry["targets"][target.name] = (
            _cache_target(config, target, all_prompts, device)
        )
        target_backbones[target.name] = backbone
        backbone_digests[target.name] = {"after_caching": _module_sha256(backbone.backbone)}

    probe = labeled["source_core_train"][0]
    results = []
    seed_seconds = {}
    _reset_peak(device)
    for seed in config.seeds:
        _progress(f"starting seed {seed}")
        started = time.perf_counter()
        results.append(
            _run_seed(
                config, seed, output_dir / f"seed-{seed}", core, core_sha256, target_backbones,
                labeled, unlabeled, teacher_outputs, representations, probe, render, device,
                transfer_train_answers_stripped,
            )
        )
        seed_seconds[str(seed)] = time.perf_counter() - started
    telemetry["seed_seconds"] = seed_seconds
    telemetry["peak_gpu_memory_mib_training"] = _peak_mib(device)

    core.assert_frozen()
    if core.state_sha256() != core_sha256:
        raise RuntimeError("the learned DecisionCore changed during the run")
    recomputed = {
        split: core_calibrated_logits(core, unlabeled[split], representations[split])
        for split in TRANSFER_SPLITS
    }
    parity = max(
        float((left - right).abs().max())
        for split in TRANSFER_SPLITS
        for left, right in zip(source_logits[split], recomputed[split], strict=True)
    )
    if parity != 0.0:
        raise RuntimeError(f"source-system outputs are not reproducible ({parity})")
    for target in config.targets:
        after = _module_sha256(target_backbones[target.name].backbone)
        backbone_digests[target.name]["after_run"] = after
        if after != backbone_digests[target.name]["after_caching"]:
            raise RuntimeError(f"the frozen {target.name} backbone changed")

    summary = aggregate_final_gate(results)
    summary["source_system"] = {
        key: source_system[key]
        for key in ("definition", "decision_core_sha256", "temperature", "metrics",
                    "openjev_head_reference_metrics")
    }
    summary["integrity"] = {
        "source_system_output_recompute_max_abs_diff": parity,
        "target_backbone_sha256": backbone_digests,
        "learned_core_sha256_after_run": core.state_sha256(),
    }
    _write_json(output_dir / "aggregate_summary.json", summary)
    _write_json(output_dir / "telemetry.json", telemetry)
    _write_json(
        output_dir / "provenance.json",
        _provenance(config, package, openjev_core, core_sha256, installed_commit, target_revisions),
    )
    return summary


def aggregate_final_gate(results: list[dict[str, object]]) -> dict[str, object]:
    """Mean and sample SD of every numeric leaf, plus paired per-seed accuracy differences."""

    if not results:
        raise ValueError("at least one result is required")
    targets = tuple(results[0]["targets"])
    if any(tuple(result["targets"]) != targets for result in results):
        raise ValueError("all seeds must contain identical targets")
    core_hashes = {result["decision_core"]["sha256"] for result in results}
    if len(core_hashes) != 1:
        raise ValueError("every seed must use the same learned decision core")
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
            "parameters": entries[0]["parameters"],
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
            "cost": {
                condition: _aggregate_trees(
                    [entry["conditions"][condition]["cost"] for entry in entries]
                )
                for condition in CONDITIONS
            },
            "comparisons": _aggregate_trees([entry["comparisons"] for entry in entries]),
            "paired_accuracy_differences": {
                split: {
                    name: _paired(
                        [
                            _accuracy(entry, "learned_core_distillation", split)
                            - _accuracy(entry, condition, split)
                            for entry in entries
                        ]
                    )
                    for condition, name in CONTROLS.items()
                }
                for split in EVALUATION_SPLITS
            },
        }
    return aggregate


def select_jevbench_seeds(results: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    """Pre-registered label-free JevBench selection (decision 0007).

    Trained conditions: the seed with the lowest final-epoch training loss for that (target,
    condition), ties to the lowest seed. The untrained adapter has no loss and uses seed 0.
    """

    ordered = sorted(results, key=lambda result: result["seed"])
    selection: dict[str, dict[str, int]] = {}
    for target in ordered[0]["targets"]:
        selection[target] = {"untrained_adapter": 0}
        for condition in TRAINED_CONDITIONS:
            final = [
                (
                    result["targets"][target]["conditions"][condition]["training"]["epoch_losses"][-1],
                    result["seed"],
                )
                for result in ordered
            ]
            selection[target][condition] = min(final)[1]
    return selection


def evaluate_ship_gate(
    results: list[dict[str, object]],
    jevbench: dict[str, dict[str, float]] | None,
    *,
    audit_passed: bool,
    reproduction_passed: bool,
) -> dict[str, object]:
    """Apply the pre-registered SHIP criteria A-E (decision 0007) mechanically.

    ``jevbench`` maps target name to the selected learned-core adapter's ``n_correct``,
    ``n_planned``, and ``uniform_expected`` on the public subset.
    """

    limits = SHIP_THRESHOLDS
    ordered = sorted(results, key=lambda result: result["seed"])
    targets = list(ordered[0]["targets"])
    seeds = len(ordered)
    need = limits["min_positive_seeds"] if seeds >= 5 else seeds

    def gains(target, control, split):
        return [
            _accuracy(result["targets"][target], "learned_core_distillation", split)
            - _accuracy(result["targets"][target], control, split)
            for result in ordered
        ]

    def mean_accuracy(target, condition, split):
        return statistics.mean(
            _accuracy(result["targets"][target], condition, split) for result in ordered
        )

    per_target: dict[str, dict[str, object]] = {}
    for target in targets:
        core_id = _paired(gains(target, "random_core_distillation", "in_distribution"))
        core_ood = _paired(gains(target, "random_core_distillation", "out_of_distribution"))
        mismatch_id = _paired(gains(target, "mismatched_teacher_distillation", "in_distribution"))
        mismatch_ood = _paired(
            gains(target, "mismatched_teacher_distillation", "out_of_distribution")
        )
        untrained_ood = _paired(gains(target, "untrained_adapter", "out_of_distribution"))
        parameters = ordered[0]["targets"][target]["parameters"]
        ratio = parameters["adapter"] / parameters["target_specific_module"]
        baseline = {}
        for split in EVALUATION_SPLITS:
            delta = mean_accuracy(target, "learned_core_distillation", split) - mean_accuracy(
                target, "target_specific_distillation", split
            )
            outperforms = delta > 0
            matches_smaller = (
                delta >= -limits["baseline_match_margin"]
                and ratio <= limits["substantially_fewer_parameter_ratio"]
            )
            baseline[split] = {
                "delta_accuracy": delta,
                "outperforms": outperforms,
                "matches_with_substantially_fewer_parameters": matches_smaller,
                "passes": outperforms or matches_smaller,
            }
        entry = {
            "A_learned_core_matters": {
                "id_gain_over_random_core": core_id,
                "ood_gain_over_random_core": core_ood,
                "passes": core_id["mean"] >= limits["core_gain_id"]
                and core_ood["mean"] >= limits["core_gain_ood"]
                and core_id["positive_seeds"] >= need
                and core_ood["positive_seeds"] >= need,
                "clearly_negative_id": core_id["mean"] < limits["clearly_negative_core_gain_id"],
            },
            "B_input_specific": {
                "id_gain_over_mismatched_teacher": mismatch_id,
                "passes": mismatch_id["mean"] >= limits["mismatch_gain_id"]
                and mismatch_id["positive_seeds"] >= need,
            },
            "C_useful_under_shift": {
                "ood_gain_over_mismatched_teacher": mismatch_ood,
                "ood_gain_over_untrained_adapter": untrained_ood,
                "passes": mismatch_ood["mean"] >= limits["ood_transfer_gain"]
                and untrained_ood["mean"] >= limits["ood_transfer_gain"]
                and mismatch_ood["positive_seeds"] >= need
                and untrained_ood["positive_seeds"] >= need,
            },
            "E_practical_utility": {
                "adapter_to_target_specific_parameter_ratio": ratio,
                "splits": baseline,
                "passes": all(value["passes"] for value in baseline.values()),
            },
        }
        if jevbench is not None and target in jevbench:
            record = jevbench[target]
            margin = (record["n_correct"] - record["uniform_expected"]) / record["n_planned"]
            entry["D_external_behavior"] = {
                **record,
                "accuracy_margin_over_uniform": margin,
                "passes": margin >= limits["jevbench_margin_over_uniform"],
            }
        else:
            entry["D_external_behavior"] = {"passes": False, "missing": True}
        per_target[target] = entry

    def count(key):
        return sum(bool(per_target[target][key]["passes"]) for target in targets)

    criteria = {
        "A_learned_core_matters": count("A_learned_core_matters") >= limits["min_targets"]
        and not any(per_target[target]["A_learned_core_matters"]["clearly_negative_id"]
                    for target in targets),
        "B_input_specific": count("B_input_specific") >= limits["min_targets"],
        "C_useful_under_shift": count("C_useful_under_shift") >= limits["min_targets"],
        "D_external_behavior": count("D_external_behavior") >= limits["min_targets"],
        "E_practical_utility": count("E_practical_utility") >= limits["min_targets"],
    }
    ship = all(criteria.values()) and audit_passed and reproduction_passed
    return {
        "verdict": "SHIP" if ship else "NO-SHIP",
        "criteria": criteria,
        "targets_passing": {
            key: [target for target in targets if per_target[target][key]["passes"]]
            for key in criteria
        },
        "audit_passed": audit_passed,
        "reproduction_passed": reproduction_passed,
        "per_target": per_target,
        "thresholds": dict(limits),
        "seeds": [result["seed"] for result in ordered],
    }


def _representation_pass(config, package, openjev_core, unlabeled, device):
    _reset_peak(device)
    started = time.perf_counter()
    teacher_device = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
    teacher = OpenJevTeacher.from_package(
        package, openjev_core, device=teacher_device, max_length=config.teacher_max_length
    )
    digest_before = teacher.adapter_and_head_sha256()
    raw, representations = {}, {}
    for split in ALL_SPLITS:
        _progress(f"representation pass: {split} ({len(unlabeled[split])} decisions)")
        raw[split], representations[split] = teacher.raw_logits_and_representations(
            unlabeled[split], candidate_batch_size=config.teacher_candidate_batch_size
        )
    digest_after = teacher.adapter_and_head_sha256()
    if digest_after != digest_before:
        raise RuntimeError("the frozen Open-Jev LoRA or head changed")
    if any(parameter.requires_grad for parameter in teacher.model.parameters()):
        raise RuntimeError("the Open-Jev source backbone is not frozen")
    telemetry = {
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": _peak_mib(device),
        "candidate_sequences": teacher.candidate_sequences,
        "input_tokens": teacher.input_tokens,
        "openjev_head_parity_max_abs_diff": teacher.parity_max_abs_diff,
        "openjev_lora_and_head_sha256_before": digest_before,
        "openjev_lora_and_head_sha256_after": digest_after,
        "source_backbone_trainable_parameters": 0,
        "representation_sha256": {
            split: _tensor_list_sha256(representations[split]) for split in ALL_SPLITS
        },
        "candidate_representations": {
            split: sum(int(block.shape[0]) for block in representations[split])
            for split in ALL_SPLITS
        },
        "max_length": config.teacher_max_length,
        "candidate_batch_size": config.teacher_candidate_batch_size,
    }
    del teacher
    _release(device)
    return raw, representations, telemetry


def _run_seed(
    config, seed, output_dir, core, core_sha256, target_backbones, labeled, unlabeled,
    teacher_outputs, representations, probe, render, device, transfer_train_answers_stripped,
):
    distillation = replace(config.distillation, seed=seed)
    random_core, matched_statistics = MLPDecisionCore.random_control(
        core,
        seed=_derived_seed(seed, "random-core"),
        representations=torch.cat(representations["source_core_train"]),
    )
    random_core = random_core.to(device)
    random_sha256 = random_core.state_sha256()
    if random_sha256 == core_sha256:
        raise RuntimeError("the random control core equals the learned core")
    random_artifact = _save_core(
        random_core,
        output_dir / "random_core",
        role="random_control_core",
        extra={"rule": RANDOM_CORE_RULE, "matched_statistics": matched_statistics},
    )
    scale = _logit_scale(core, random_core, unlabeled["train"], representations["train"])
    # The broad train split has one 5-option Choice decision; it cannot be deranged within its
    # kind and width, so it keeps its own teacher output and is reported as a fixed point.
    mismatched = permute_teacher_outputs(
        unlabeled["train"],
        teacher_outputs["train"],
        seed=seed,
        group_key=_typed_group,
        keep_singletons=True,
    )
    fixed_points = [
        index
        for index, (original, control) in enumerate(
            zip(teacher_outputs["train"], mismatched, strict=True)
        )
        if original is control
    ]
    identity = ScalarIdentityCore(core.temperature).to(device)
    targets = {}
    for target in config.targets:
        _progress(f"seed {seed}: target {target.name}")
        targets[target.name] = _run_target(
            config, target, seed, output_dir / "targets" / target.name, core, random_core,
            identity, target_backbones[target.name], labeled, unlabeled, teacher_outputs,
            mismatched, distillation, probe, render, device,
        )
    for frozen, digest in ((core, core_sha256), (random_core, random_sha256)):
        frozen.assert_frozen()
        if frozen.state_sha256() != digest:
            raise RuntimeError("a frozen decision core changed")
    result = {
        "status": STATUS_SMOKE if config.is_smoke else STATUS_FULL,
        "seed": seed,
        "config": _config_to_dict(config, seed),
        "decision_core": {
            "kind": "learned_mlp_decision_core",
            "architecture": _architecture(core.input_size, config.core_hidden_sizes),
            "sha256": core_sha256,
            "input_size": core.input_size,
            "parameters": core.parameter_count,
            "temperature": core.temperature,
            "shared_by_targets": [target.name for target in config.targets],
        },
        "random_control_core": {
            "sha256": random_sha256,
            "rule": RANDOM_CORE_RULE,
            "derived_seed": _derived_seed(seed, "random-core"),
            "matching_representations": "source_core_train (all candidates, unlabeled)",
            "matched_statistics": matched_statistics,
            "artifact": random_artifact,
            "raw_candidate_score_std_on_train_inputs": scale,
        },
        "mismatched_teacher_control": {
            "grouping": "decision kind and option count",
            "decisions": len(mismatched),
            "fixed_point_decisions": len(fixed_points),
            "fixed_point_groups": sorted(
                {"{}:{}".format(*_typed_group(unlabeled["train"][index])) for index in fixed_points}
            ),
            "fixed_point_reason": "singleton kind-and-width group cannot be deranged",
        },
        "targets": targets,
        "label_free_audit": {
            "transfer_entry_point_rejects_labeled_records": True,
            "guard_probe": "labeled source_core_train record",
            "transfer_train_answers_stripped_at_load": transfer_train_answers_stripped,
            "distillation_decisions": len(unlabeled["train"]),
            "distillation_decisions_with_answers": sum(
                item.answer is not None for item in unlabeled["train"]
            ),
            "teacher_inputs_with_answers": sum(
                item.answer is not None for split in ALL_SPLITS for item in unlabeled[split]
            ),
            "source_core_labeled_decisions": len(labeled["source_core_train"]),
            "source_core_calibration_labeled_decisions": len(labeled["source_core_calibration"]),
            "source_core_records_overlapping_transfer_splits": 0,
            "ground_truth_used_for_target_distillation": False,
            "label_based_early_stopping_selection_weighting_or_tuning": False,
            "teacher_outputs_detached": all(
                not output.scores.requires_grad
                for outputs in teacher_outputs.values()
                for output in outputs
            ),
            "optimized_components": [
                f"{target.name}.{condition}.{_module_role(condition)}"
                for target in config.targets
                for condition in TRAINED_CONDITIONS
            ],
            "frozen_components": [
                "openjev_qwen3.5_2b_backbone",
                "openjev_lora_adapter",
                "learned_decision_core",
                "learned_decision_core_calibration",
                "random_control_core",
                *[f"{target.name}_backbone" for target in config.targets],
            ],
            "same_learned_core_sha256_for_all_targets": core_sha256,
            "learned_core_verified_unchanged": True,
        },
    }
    _write_json(output_dir / "results.json", result)
    return result


def _run_target(
    config, target, seed, output_dir, core, random_core, identity, backbone, labeled, unlabeled,
    teacher_outputs, mismatched, distillation, probe, render, device,
):
    torch.manual_seed(_derived_seed(seed, target.name))
    initial = LowRankAdapter(backbone.hidden_size, core.input_size, rank=config.adapter_rank)
    initial_state = _state_copy(initial)
    torch.manual_seed(_derived_seed(seed, f"{target.name}:target-specific"))
    baseline = TargetDecisionModule(backbone.hidden_size, config.baseline_hidden_sizes)
    baseline_state = _state_copy(baseline)
    conditions = {}

    untrained = _adapter_model(backbone, initial_state, core, config, device)
    start = _state_sha256(_state_copy(untrained.adapter))
    _assert_guard_rejects_labels(untrained, probe, render)
    conditions["untrained_adapter"] = _record(
        untrained, "untrained_adapter", output_dir, labeled, unlabeled, teacher_outputs, config,
        distillation, render, None, core_role="learned_core", initial_state_sha256=start,
    )
    for condition, condition_core, core_role, objective, teacher_kind in (
        ("learned_core_distillation", core, "learned_core", teacher_outputs["train"], "matched"),
        (
            "mismatched_teacher_distillation", core, "learned_core", mismatched,
            "deranged_within_type_and_width",
        ),
        ("random_core_distillation", random_core, "random_core", teacher_outputs["train"],
         "matched"),
    ):
        model = _adapter_model(backbone, initial_state, condition_core, config, device)
        start = _state_sha256(_state_copy(model.adapter))
        training = _train(model, unlabeled, objective, distillation, render, config, teacher_kind)
        conditions[condition] = _record(
            model, condition, output_dir, labeled, unlabeled, teacher_outputs, config,
            distillation, render, training, core_role=core_role, initial_state_sha256=start,
        )
    module = TargetDecisionModule(backbone.hidden_size, config.baseline_hidden_sizes).to(device)
    module.load_state_dict(baseline_state)
    model = DecPort(backbone, module, identity)
    start = _state_sha256(_state_copy(model.adapter))
    training = _train(
        model, unlabeled, teacher_outputs["train"], distillation, render, config, "matched"
    )
    conditions["target_specific_distillation"] = _record(
        model, "target_specific_distillation", output_dir, labeled, unlabeled, teacher_outputs,
        config, distillation, render, training, core_role="none", initial_state_sha256=start,
    )
    initial_hashes = {condition: conditions[condition]["initial_state_sha256"]
                      for condition in ADAPTER_CONDITIONS}
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("adapter conditions did not start from one matched state")
    return {
        "backbone_type": target.backbone_type,
        "model_id": target.model_id,
        "parameters": {
            "backbone_hidden_size": backbone.hidden_size,
            "adapter": initial.parameter_count,
            "adapter_rank": initial.rank,
            "target_specific_module": baseline.parameter_count,
            "adapter_to_target_specific_ratio": initial.parameter_count / baseline.parameter_count,
            "shared_core_not_per_target": core.parameter_count,
            "multiply_accumulates_per_candidate": {
                "adapter": backbone.hidden_size * config.adapter_rank
                + config.adapter_rank * core.input_size,
                "shared_core": _mlp_macs(core.input_size, config.core_hidden_sizes),
                "target_specific_module": _mlp_macs(
                    backbone.hidden_size, config.baseline_hidden_sizes
                ),
            },
        },
        "matched_initialization": {
            "adapter_sha256": _state_sha256(initial_state),
            "adapter_derived_seed": _derived_seed(seed, target.name),
            "target_specific_sha256": _state_sha256(baseline_state),
            "target_specific_derived_seed": _derived_seed(seed, f"{target.name}:target-specific"),
            "adapter_conditions_share_initial_state": True,
        },
        "conditions": conditions,
        "comparisons": {split: _comparisons(conditions, split) for split in EVALUATION_SPLITS},
    }


def _train(model, unlabeled, objective, distillation, render, config, teacher_kind):
    started = time.perf_counter()
    history = train_core_distillation_batched(
        model,
        unlabeled["train"],
        objective,
        distillation,
        render=render,
        batch_size=config.train_batch_size,
    )
    return {
        "teacher_outputs": teacher_kind,
        "seconds": time.perf_counter() - started,
        "epoch_losses": list(history.epoch_losses),
        "epoch_kl_divergence": list(history.epoch_kl_divergence),
        "epoch_centered_logit_mse": list(history.epoch_centered_logit_mse),
        "gradient_audit": history.gradient_audit,
    }


def _record(
    model, condition, output_dir, labeled, unlabeled, teacher_outputs, config, distillation,
    render, training, *, core_role, initial_state_sha256,
):
    artifact = _save_module(model, condition, output_dir / condition, core_role)
    logits, seconds = {}, {}
    for split in TRANSFER_SPLITS:
        started = time.perf_counter()
        logits[split] = predict_core_logits(
            model, unlabeled[split], render=render, batch_size=config.eval_batch_size
        )
        seconds[split] = time.perf_counter() - started
    return {
        "training": training,
        "initial_state_sha256": initial_state_sha256,
        "metrics": {
            split: evaluate_typed(labeled[split], logits[split]) for split in EVALUATION_SPLITS
        },
        "decision_match": {
            split: typed_match(
                unlabeled[split], logits[split], teacher_outputs[split], distillation
            )
            for split in TRANSFER_SPLITS
        },
        "cost": {
            "trainable_parameters": 0 if training is None else model.adapter.parameter_count,
            "stored_parameters": model.adapter.parameter_count,
            "artifact_bytes": artifact["bytes"],
            "training_seconds": 0.0 if training is None else training["seconds"],
            "inference_seconds_from_cached_states": seconds,
        },
        "artifact": artifact,
    }


def _save_module(model, condition, directory, core_role):
    """Save the per-target trainable module; the shared/random cores are saved once elsewhere."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    module = model.adapter
    is_adapter = isinstance(module, LowRankAdapter)
    filename = "adapter.safetensors" if is_adapter else "module.safetensors"
    state = {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}
    save_file(state, directory / filename)
    config = {
        "format": "decport.final_gate.v0.1",
        "role": condition,
        "module": "LowRankAdapter" if is_adapter else "TargetDecisionModule",
        "input_size": module.input_size,
        "output_size": module.shared_size,
        "rank": module.rank if is_adapter else None,
        "hidden_sizes": None if is_adapter else list(module.hidden_sizes),
        "parameters": module.parameter_count,
        "decision_core": core_role,
        "decision_core_sha256": model.head.state_sha256(),
        "decision_core_temperature": model.head.temperature,
        "decision_labels_used": False,
        "backbone_model_id": getattr(model.backbone, "model_id", None),
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "file": filename,
        "sha256": _file_sha256(directory / filename),
        "bytes": (directory / filename).stat().st_size,
    }


def _save_core(core, directory, *, role, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous() for key, value in core.network.state_dict().items()
    }
    save_file(state, directory / "core.safetensors")
    config = {
        "format": "decport.final_gate.v0.1",
        "role": role,
        "architecture": _architecture(core.input_size, core.hidden_sizes),
        "input_size": core.input_size,
        "hidden_sizes": list(core.hidden_sizes),
        "temperature": core.temperature,
        "parameters": core.parameter_count,
        "state_sha256": core.state_sha256(),
        **(extra or {}),
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": str(directory),
        "sha256": _file_sha256(directory / "core.safetensors"),
        "bytes": (directory / "core.safetensors").stat().st_size,
        "state_sha256": core.state_sha256(),
    }


def _adapter_model(backbone, adapter_state, core, config, device):
    adapter = LowRankAdapter(backbone.hidden_size, core.input_size, rank=config.adapter_rank)
    adapter = adapter.to(device)
    adapter.load_state_dict(adapter_state)
    return DecPort(backbone, adapter, core)


def _comparisons(conditions, split):
    """Learned-core gains over each control: accuracy, teacher agreement, and teacher KL."""

    def gains(select):
        learned = select(conditions["learned_core_distillation"])
        values = {
            "learned_core_accuracy": float(learned[0]),
            "learned_core_agreement": float(learned[1]),
            "learned_core_teacher_kl": float(learned[2]),
        }
        for condition, name in CONTROLS.items():
            control = select(conditions[condition])
            values[f"{name}_accuracy"] = float(control[0])
            values[f"{name}_agreement"] = float(control[1])
            values[f"{name}_teacher_kl"] = float(control[2])
            values[f"accuracy_gain_over_{name}"] = float(learned[0] - control[0])
            values[f"agreement_gain_over_{name}"] = float(learned[1] - control[1])
            values[f"teacher_kl_reduction_over_{name}"] = float(control[2] - learned[2])
        return values

    def pick(group, key=None):
        def select(condition):
            metrics = condition["metrics"][split][group]
            match = condition["decision_match"][split][group]
            if key is not None:
                metrics, match = metrics[key], match[key]
            return metrics["accuracy"], match["top_choice_agreement"], match["kl_divergence"]

        return select

    reference = conditions["learned_core_distillation"]["metrics"][split]
    return {
        "overall": gains(pick("overall")),
        "macro_across_decision_types": gains(pick("macro_across_decision_types")),
        "by_decision_type": {
            kind: gains(pick("by_decision_type", kind)) for kind in reference["by_decision_type"]
        },
        "by_dataset": {name: gains(pick("by_dataset", name)) for name in reference["by_dataset"]},
    }


def _assert_guard_rejects_labels(model, labeled_decision, render):
    """Exercise the hard label guard on the real transfer entry point before any training."""

    try:
        train_core_distillation_batched(
            model,
            [labeled_decision],
            (TeacherOutput(scores=torch.zeros(len(labeled_decision.options))),),
            DistillationConfig(epochs=1),
            render=render,
            batch_size=1,
        )
    except ValueError as error:
        if "must not include answers" in str(error):
            return
        raise
    raise RuntimeError("the transfer entry point accepted a labeled record")


def _validate_config(config):
    sizes = (
        config.adapter_rank,
        *config.core_hidden_sizes,
        *config.baseline_hidden_sizes,
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
        set(config.limit_per_type) != set(ALL_SPLITS)
        or min(config.limit_per_type.values()) < 2
    ):
        raise ValueError("limit_per_type needs every split and at least two decisions per type")


def _validate_data(splits, *, bounded):
    for split, decisions in splits.items():
        if {decision.kind for decision in decisions} != set(DECISION_KINDS):
            raise ValueError(f"{split} must contain Choice, Noul, and Score decisions")
        if any(not decision.dataset for decision in decisions):
            raise ValueError(f"{split} needs dataset metadata")
        if split != "train" and any(decision.answer is None for decision in decisions):
            raise ValueError(f"{split} needs labels (source training or evaluation)")
        if any(
            decision.kind == "score" and decision.criteria != REVIEW_LEVELS
            for decision in decisions
        ):
            raise ValueError("Score levels must be the ordered five-level review scale")
        if not bounded and len(decisions) != FULL_SPLIT_COUNTS[split]:
            raise ValueError(f"{split} must contain {FULL_SPLIT_COUNTS[split]} decisions")


def _content_overlap(left, right) -> int:
    def signature(decision):
        return hashlib.sha256(
            json.dumps(
                {"state": decision.state, "question": decision.question()},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    return len({signature(item) for item in left} & {signature(item) for item in right})


def _logit_scale(core, random_core, decisions, representations):
    hidden = torch.cat(representations)
    with torch.inference_mode():
        learned = torch.cat([core(chunk) for chunk in hidden.split(4096)])
        random_scores = torch.cat([random_core(chunk) for chunk in hidden.split(4096)])
    return {
        "learned_core": float(learned.float().std()),
        "random_core": float(random_scores.float().std()),
        "candidates": int(hidden.shape[0]),
        "decisions": len(decisions),
    }


def _accuracy(target_entry, condition, split):
    return float(target_entry["conditions"][condition]["metrics"][split]["overall"]["accuracy"])


def _paired(values):
    values = [float(value) for value in values]
    return {
        "per_seed": values,
        "mean": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_seeds": sum(value > 0 for value in values),
        "seeds": len(values),
    }


def _module_role(condition):
    return "target_specific_module" if condition == "target_specific_distillation" else "adapter"


def _architecture(input_size, hidden_sizes):
    first, second = hidden_sizes
    return (
        f"LayerNorm({input_size}) -> Linear({input_size},{first}) -> GELU -> "
        f"Linear({first},{second}) -> GELU -> Linear({second},1)"
    )


def _mlp_macs(input_size, hidden_sizes):
    first, second = hidden_sizes
    return input_size * first + first * second + second


def _module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _tensor_list_sha256(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _provenance(config, package, openjev_core, core_sha256, installed_commit, target_revisions):
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
            "openjev_head_sha256": openjev_core.state_sha256(),
        },
        "learned_decision_core_sha256": core_sha256,
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
                ("source_core_train", config.source_train_path),
                ("source_core_calibration", config.source_calibration_path),
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


def _config_to_dict(config, seed):
    value = asdict(config)
    value["seed"] = seed
    return value


def _progress(message):
    print(message, file=sys.stderr, flush=True)
