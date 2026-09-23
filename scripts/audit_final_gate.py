#!/usr/bin/env python3
"""Audit completeness and every frozen-protocol invariant of a final-gate run (decision 0007)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = (
    "untrained_adapter",
    "learned_core_distillation",
    "mismatched_teacher_distillation",
    "random_core_distillation",
    "target_specific_distillation",
)
ADAPTER_CONDITIONS = CONDITIONS[:4]
TRAINED = CONDITIONS[1:]
TARGETS = {"smollm", "gemma", "tinyllama"}
SEEDS = [0, 1, 2, 3, 4]
KINDS = {"choice", "noul", "score"}
MANIFEST_KINDS = {"choice": "choice", "boolean": "noul", "score": "score"}
TRANSFER_FILES = {
    "train": "train.jsonl",
    "in_distribution": "eval.jsonl",
    "out_of_distribution": "ood.jsonl",
}
SOURCE_FILES = {
    "source_core_train": "source_train.jsonl",
    "source_core_calibration": "source_calibration.jsonl",
}
EVALUATION_SPLITS = ("in_distribution", "out_of_distribution")
PARITY_TOLERANCE = 1e-5
EPOCHS = 10
EXPECTED_FIXED_POINTS = 1
EXPECTED_FIXED_POINT_GROUPS = ["choice:5"]
EXPECTED_PARAMETERS = {
    "smollm": {"adapter": 386_944, "target_specific_module": 559_745},
    "gemma": {"adapter": 345_344, "target_specific_module": 395_265},
    "tinyllama": {"adapter": 528_384, "target_specific_module": 1_118_977},
}
CORE_PARAMETERS = 1_118_977
# LowRankAdapter: LayerNorm weight/bias + two bias-free linears. Target module: LN + 3 linears.
TRAINABLE_TENSORS = {"adapter": 4, "target_specific_module": 8}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--config", required=True, help="the frozen repository config")
    parser.add_argument(
        "--verify-core",
        action="store_true",
        help="reload the learned core and every random core from their artifacts and compare "
        "digests (needs the decport package)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="audit a bounded smoke run: its seeds, slices, and epochs come from its config",
    )
    args = parser.parse_args()
    root, data, source = Path(args.run), Path(args.data), Path(args.source_data)
    summary = _json(root / "aggregate_summary.json")
    provenance = _json(root / "provenance.json")
    telemetry = _json(root / "telemetry.json")
    experiment = _json(root / "experiment_config.json")
    frozen = _json(Path(args.config))
    seeds = frozen["seeds"] if args.smoke else SEEDS
    epochs = frozen["epochs"] if args.smoke else EPOCHS
    limits = frozen.get("limit_per_type") if args.smoke else None
    _check(summary["seeds"] == seeds and summary["seed_count"] == len(seeds), "seed list")
    _check(set(summary["targets"]) == TARGETS, "expected all three targets")
    _check(
        summary["status"]
        == ("smoke_test_not_evidence" if args.smoke else "final_gate_pending_verdict"),
        "unexpected run status",
    )
    _check(
        {key: experiment[key] for key in frozen} == frozen,
        "the run's configuration differs from the frozen repository config",
    )
    _check(
        _sha256(root / "repository_config.json") == _sha256(Path(args.config)),
        "archived repository config differs from the frozen config",
    )
    _assert_finite(summary)

    openjev = provenance["openjev"]
    _check(openjev["installed_commit"] == openjev["pinned_commit"], "Open-Jev commit mismatch")
    _check(
        openjev["package"]["manifest_sha256"] == frozen["openjev_package_manifest_sha256"],
        "Open-Jev package manifest mismatch",
    )
    counts = {}
    for directory, files in ((data, TRANSFER_FILES), (source, SOURCE_FILES)):
        manifest = _json(directory / "data_manifest.json")
        for split, filename in files.items():
            digest = _sha256(directory / filename)
            _check(digest == manifest["splits"][split]["sha256"], f"{split} data checksum")
            _check(provenance["data"][split]["sha256"] == digest, f"{split} provenance hash")
            counts[split] = _file_counts(directory / filename, limits and limits[split])
            if not args.smoke:
                expected = _expected_counts(manifest["splits"][split], split)
                _check(
                    {key: counts[split][key] for key in expected} == expected,
                    f"{split} composition differs from its manifest",
                )

    teacher = telemetry["teacher"]
    _check(
        teacher["openjev_head_parity_max_abs_diff"] <= PARITY_TOLERANCE,
        "Open-Jev head / captured representation parity failed",
    )
    _check(
        teacher["openjev_lora_and_head_sha256_before"]
        == teacher["openjev_lora_and_head_sha256_after"],
        "the frozen Open-Jev LoRA or head changed",
    )
    _check(teacher["source_backbone_trainable_parameters"] == 0, "source backbone not frozen")
    for split in counts:
        _check(
            teacher["candidate_representations"][split] == counts[split]["candidates"],
            f"{split} representation count",
        )

    core_sha256 = summary["decision_core_sha256"]
    _check(provenance["learned_decision_core_sha256"] == core_sha256, "core digest provenance")
    integrity = summary["integrity"]
    _check(integrity["learned_core_sha256_after_run"] == core_sha256, "core changed after run")
    _check(
        integrity["source_system_output_recompute_max_abs_diff"] == 0.0,
        "source-system outputs are not reproducible",
    )
    for name, digests in integrity["target_backbone_sha256"].items():
        _check(digests["after_caching"] == digests["after_run"], f"{name} backbone changed")
    core_config = _json(root / "source_core" / "config.json")
    _check(core_config["state_sha256"] == core_sha256, "core artifact digest")
    _check(core_config["parameters"] == CORE_PARAMETERS, "core parameter count")
    _check(core_config["hidden_sizes"] == frozen["core_hidden_sizes"], "core architecture")
    _check(len(core_config["training"]["epoch_losses"]) == frozen["source_core"]["epochs"],
           "source-core epochs")
    _check(
        core_config["training"]["labeled_source_decisions"]
        == counts["source_core_train"]["total"],
        "source-core training size",
    )
    source_system = _json(root / "source_system" / "summary.json")
    for split in EVALUATION_SPLITS:
        _check_metric_counts(source_system["metrics"][split], counts[split], f"source {split}")
    logits = _json(root / "source_system" / "teacher_logits.json")
    for split in TRANSFER_FILES:
        _check(len(logits[split]) == counts[split]["total"], f"teacher {split} rows")

    random_hashes = set()
    for seed in seeds:
        seed_root = root / f"seed-{seed}"
        result = _json(seed_root / "results.json")
        _assert_finite(result)
        _check(result["seed"] == seed, f"seed {seed} mislabeled")
        audit = result["label_free_audit"]
        label = f"seed {seed}"
        _check(audit["transfer_entry_point_rejects_labeled_records"], f"{label} label guard")
        _check(audit["distillation_decisions"] == counts["train"]["total"], f"{label} train size")
        _check(audit["distillation_decisions_with_answers"] == 0, f"{label} labeled records")
        _check(audit["teacher_inputs_with_answers"] == 0, f"{label} labeled teacher inputs")
        _check(
            audit["transfer_train_answers_stripped_at_load"] == counts["train"]["total"],
            f"{label} transfer-train answers were not stripped at load",
        )
        _check(audit["source_core_records_overlapping_transfer_splits"] == 0, f"{label} overlap")
        _check(not audit["ground_truth_used_for_target_distillation"], f"{label} used labels")
        _check(
            not audit["label_based_early_stopping_selection_weighting_or_tuning"],
            f"{label} label-based selection",
        )
        _check(audit["teacher_outputs_detached"], f"{label} attached teacher outputs")
        _check(audit["learned_core_verified_unchanged"], f"{label} core changed")
        _check(
            audit["same_learned_core_sha256_for_all_targets"] == core_sha256
            and result["decision_core"]["sha256"] == core_sha256,
            f"{label} used a different core",
        )
        _check(
            set(result["decision_core"]["shared_by_targets"]) == TARGETS,
            f"{label} core not shared by every target",
        )
        control = result["mismatched_teacher_control"]
        _check(control["decisions"] == counts["train"]["total"], f"{label} mismatched size")
        if not args.smoke:
            _check(
                control["fixed_point_decisions"] == EXPECTED_FIXED_POINTS
                and control["fixed_point_groups"] == EXPECTED_FIXED_POINT_GROUPS,
                f"{label} mismatched teacher has unexpected fixed points {control}",
            )
        random_core = result["random_control_core"]
        random_sha256 = random_core["sha256"]
        _check(random_sha256 != core_sha256, f"{label} random core equals the learned core")
        _check(
            _sha256(seed_root / "random_core" / "core.safetensors")
            == random_core["artifact"]["sha256"],
            f"{label} random core artifact",
        )
        random_hashes.add(random_sha256)
        for name in TARGETS:
            target = result["targets"][name]
            _check(set(target["conditions"]) == set(CONDITIONS), f"{label} {name} conditions")
            parameters = target["parameters"]
            for key, value in EXPECTED_PARAMETERS[name].items():
                _check(parameters[key] == value, f"{label} {name} {key} parameters")
            _check(parameters["adapter_rank"] == frozen["adapter_rank"], f"{label} {name} rank")
            matched = target["matched_initialization"]
            for condition in CONDITIONS:
                where = f"{label} {name} {condition}"
                record = target["conditions"][condition]
                artifact = seed_root / "targets" / name / condition
                config = _json(artifact / "config.json")
                file = artifact / record["artifact"]["file"]
                _check(_sha256(file) == record["artifact"]["sha256"], f"{where} artifact hash")
                _check(config["decision_labels_used"] is False, f"{where} labels")
                expected_core = {
                    "random_core_distillation": random_sha256,
                    "target_specific_distillation": None,
                }.get(condition, core_sha256)
                if expected_core is None:
                    _check(config["decision_core"] == "none", f"{where} reused a shared core")
                    _check(config["module"] == "TargetDecisionModule", f"{where} module")
                    _check(
                        record["initial_state_sha256"] == matched["target_specific_sha256"],
                        f"{where} initialization",
                    )
                    role = "target_specific_module"
                else:
                    _check(config["decision_core_sha256"] == expected_core, f"{where} core")
                    _check(config["module"] == "LowRankAdapter", f"{where} module")
                    _check(
                        record["initial_state_sha256"] == matched["adapter_sha256"],
                        f"{where} matched initialization",
                    )
                    role = "adapter"
                training = record["training"]
                if condition in TRAINED:
                    gradients = training["gradient_audit"]
                    _check(
                        gradients["adapter_tensors"] == TRAINABLE_TENSORS[role]
                        and gradients["adapter_tensors_with_gradient"] == TRAINABLE_TENSORS[role],
                        f"{where} trainable gradients",
                    )
                    _check(gradients["core_tensors_with_gradient"] == 0, f"{where} core grads")
                    _check(gradients["backbone_tensors_with_gradient"] == 0, f"{where} LM grads")
                    _check(len(training["epoch_losses"]) == epochs, f"{where} epochs")
                    expected_teacher = (
                        "deranged_within_type_and_width"
                        if condition == "mismatched_teacher_distillation"
                        else "matched"
                    )
                    _check(training["teacher_outputs"] == expected_teacher, f"{where} teacher")
                else:
                    _check(training is None, f"{where} untrained adapter was trained")
                for split in EVALUATION_SPLITS:
                    _check_metric_counts(
                        record["metrics"][split], counts[split], f"{where} {split}"
                    )
                for split in TRANSFER_FILES:
                    match = record["decision_match"][split]
                    _check(
                        match["overall"]["example_count"] == counts[split]["total"]
                        and set(match["by_decision_type"]) == KINDS,
                        f"{where} {split} teacher-match coverage",
                    )
    _check(len(random_hashes) == len(seeds), "random control cores are not seed-specific")

    for name in TARGETS:
        metrics = summary["targets"][name]["metrics"]
        for condition in CONDITIONS:
            for split in EVALUATION_SPLITS:
                by_type = metrics[condition][split]["by_decision_type"]
                _check(set(by_type) == KINDS, f"aggregate {name} {condition} {split} types")

    reloaded = None
    if args.verify_core:
        from safetensors.torch import load_file

        from decport.decision_core import MLPDecisionCore

        state = load_file(root / "source_core" / "core.safetensors")
        core = MLPDecisionCore(
            core_config["input_size"],
            core_config["temperature"],
            state=state,
            hidden_sizes=tuple(core_config["hidden_sizes"]),
        )
        reloaded = core.state_sha256()
        _check(reloaded == core_sha256, "the reloaded learned core differs from the run's core")
        for seed in seeds:
            random_config = _json(root / f"seed-{seed}" / "random_core" / "config.json")
            random_core = MLPDecisionCore(
                random_config["input_size"],
                random_config["temperature"],
                state=load_file(root / f"seed-{seed}" / "random_core" / "core.safetensors"),
                hidden_sizes=tuple(random_config["hidden_sizes"]),
            )
            _check(
                random_core.state_sha256() == random_config["state_sha256"],
                f"seed {seed} reloaded random core differs",
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "smoke_run_not_evidence": args.smoke,
                "seeds": seeds,
                "targets": sorted(TARGETS),
                "conditions_per_target": list(CONDITIONS),
                "learned_decision_core_sha256": core_sha256,
                "learned_core_reloaded_from_artifact_sha256": reloaded,
                "openjev_head_parity_max_abs_diff": teacher["openjev_head_parity_max_abs_diff"],
                "source_system_output_recompute_max_abs_diff": integrity[
                    "source_system_output_recompute_max_abs_diff"
                ],
                "mismatched_teacher_fixed_points_per_seed": EXPECTED_FIXED_POINTS,
                "frozen_config_matches_run": True,
                "data_hashes_and_composition_verified": True,
                "source_core_disjoint_from_transfer_splits": True,
                "every_decision_counted_in_every_metric": True,
                "gradients_only_in_allowed_tensors": True,
                "matched_adapter_initialization_verified": True,
                "frozen_source_and_target_backbones_verified": True,
                "strict_label_free_boundary_verified": True,
                "artifact_hashes_verified": True,
                "all_recorded_numbers_finite": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _expected_counts(record, split):
    by_kind = {MANIFEST_KINDS[kind]: count for kind, count in record["decision_types"].items()}
    _check(set(by_kind) == KINDS, f"{split} manifest decision types")
    return {"total": record["count"], "kinds": by_kind, "datasets": record["datasets"]}


def _file_counts(path: Path, limit: int | None) -> dict[str, object]:
    """Decisions, per-type/per-dataset counts, and scored candidates, with an optional smoke
    limit of the first N decisions of each type (the runner's slicing rule)."""

    kinds: dict[str, int] = {}
    datasets: dict[str, int] = {}
    candidates = total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = MANIFEST_KINDS[record["decision_type"]]
            if limit is not None and kinds.get(kind, 0) >= limit:
                continue
            kinds[kind] = kinds.get(kind, 0) + 1
            datasets[record["dataset"]] = datasets.get(record["dataset"], 0) + 1
            candidates += 1 if kind == "noul" else len(record["options"])
            total += 1
    return {
        "total": total,
        "kinds": kinds,
        "datasets": dict(sorted(datasets.items())),
        "candidates": candidates,
    }


def _check_metric_counts(metrics, expected, label):
    """No decision may be dropped: every overall/type/dataset count must equal the data."""

    _check(metrics["overall"]["example_count"] == expected["total"], f"{label} overall count")
    by_type = {kind: value["example_count"] for kind, value in metrics["by_decision_type"].items()}
    _check(by_type == expected["kinds"], f"{label} per-type counts {by_type}")
    by_dataset = {name: value["example_count"] for name, value in metrics["by_dataset"].items()}
    _check(by_dataset == expected["datasets"], f"{label} per-dataset counts {by_dataset}")


def _check(condition, message):
    if not condition:
        raise RuntimeError(f"audit failed: {message}")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("audit failed: result contains a non-finite number")


if __name__ == "__main__":
    main()
