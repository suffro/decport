#!/usr/bin/env python3
"""Audit completeness and strict transfer invariants of an Open-Jev DecisionCore transfer run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = (
    "untrained_target_adapter",
    "openjev_teacher_distillation",
    "mismatched_teacher_distillation",
    "random_core_distillation",
)
TRAINED = CONDITIONS[1:]
TARGETS = {"smollm", "gemma", "tinyllama"}
SEEDS = [0, 1, 2, 3, 4]
KINDS = {"choice", "noul", "score"}
MANIFEST_KINDS = {"choice": "choice", "boolean": "noul", "score": "score"}
SPLITS = {
    "train": "train.jsonl",
    "in_distribution": "eval.jsonl",
    "out_of_distribution": "ood.jsonl",
}
EVALUATION_SPLITS = ("in_distribution", "out_of_distribution")
PARITY_TOLERANCE = 1e-5
# jev-broad-v0.1 train has exactly one 5-option Choice decision, the only singleton
# kind-and-width group; every other mismatched-teacher output must be deranged.
EXPECTED_FIXED_POINTS = 1
EXPECTED_FIXED_POINT_GROUPS = ["choice:5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--verify-core",
        action="store_true",
        help="independently reload the pinned Open-Jev core and compare its digest (needs the "
        "openjev extra and the cached package)",
    )
    args = parser.parse_args()
    root = Path(args.run)
    data = Path(args.data)
    summary = _json(root / "aggregate_summary.json")
    provenance = _json(root / "provenance.json")
    telemetry = _json(root / "telemetry.json")
    manifest = _json(data / "data_manifest.json")
    _check(summary["seeds"] == SEEDS and summary["seed_count"] == 5, "expected seeds 0-4")
    _check(set(summary["targets"]) == TARGETS, "expected all three targets")
    _check(summary["status"] == "openjev_transfer_pending_interpretation", "not a full run")
    _assert_finite(summary)

    core_sha256 = provenance["openjev"]["decision_core_sha256"]
    _check(summary["decision_core_sha256"] == core_sha256, "aggregate core digest mismatch")
    openjev = provenance["openjev"]
    _check(openjev["installed_commit"] == openjev["pinned_commit"], "Open-Jev commit mismatch")
    _check(
        openjev["package"]["manifest_sha256"]
        == _json(root / "experiment_config.json")["openjev_package_manifest_sha256"],
        "Open-Jev package manifest mismatch",
    )

    counts = _expected_counts(manifest)
    for split, filename in SPLITS.items():
        digest = _sha256(data / filename)
        _check(digest == manifest["splits"][split]["sha256"], f"{split} data checksum mismatch")
        _check(provenance["data"][split]["sha256"] == digest, f"{split} provenance mismatch")

    teacher = telemetry["teacher"]
    _check(teacher["core_parity_max_abs_diff"] <= PARITY_TOLERANCE, "teacher/core parity failed")
    raw = _json(root / "teacher" / "raw_logits.json")
    for split in SPLITS:
        _check(len(raw[split]) == counts[split]["total"], f"teacher {split} rows missing")
    teacher_summary = _json(root / "teacher" / "summary.json")
    for split in EVALUATION_SPLITS:
        _check_metric_counts(teacher_summary["metrics"][split], counts[split], f"teacher {split}")

    random_hashes = set()
    for seed in SEEDS:
        seed_root = root / f"seed-{seed}"
        result = _json(seed_root / "results.json")
        _assert_finite(result)
        _check(result["seed"] == seed, f"seed {seed} mislabeled")
        audit = result["label_free_audit"]
        _check(audit["transfer_entry_point_rejects_labeled_records"], f"seed {seed} label guard")
        _check(audit["distillation_decisions"] == counts["train"]["total"], f"seed {seed} train")
        _check(audit["distillation_decisions_with_answers"] == 0, f"seed {seed} labeled records")
        _check(audit["teacher_inputs_with_answers"] == 0, f"seed {seed} labeled teacher inputs")
        _check(not audit["ground_truth_used_for_target_distillation"], f"seed {seed} used labels")
        _check(
            not audit["label_based_early_stopping_selection_weighting_or_tuning"],
            f"seed {seed} label-based selection",
        )
        _check(audit["teacher_outputs_detached"], f"seed {seed} attached teacher outputs")
        _check(audit["openjev_core_verified_unchanged"], f"seed {seed} core changed")
        _check(
            audit["same_openjev_core_sha256_for_all_targets"] == core_sha256
            and result["decision_core"]["sha256"] == core_sha256,
            f"seed {seed} used a different core",
        )
        _check(
            set(result["decision_core"]["shared_by_targets"]) == TARGETS,
            f"seed {seed} core not shared by every target",
        )
        control = result["mismatched_teacher_control"]
        _check(
            control["decisions"] == counts["train"]["total"]
            and control["fixed_point_decisions"] == EXPECTED_FIXED_POINTS
            and control["fixed_point_groups"] == EXPECTED_FIXED_POINT_GROUPS,
            f"seed {seed} mismatched teacher has unexpected fixed points {control}",
        )
        random_sha256 = result["random_control_core_sha256"]
        _check(random_sha256 != core_sha256, f"seed {seed} random core equals Open-Jev core")
        random_hashes.add(random_sha256)
        for name in TARGETS:
            target = result["targets"][name]
            _check(tuple(target["conditions"]) == CONDITIONS, f"seed {seed} {name} conditions")
            for condition in CONDITIONS:
                label = f"seed {seed} {name} {condition}"
                record = target["conditions"][condition]
                artifact = seed_root / "targets" / name / condition
                config = _json(artifact / "config.json")
                metadata = config["metadata"]
                _check((artifact / "adapter.safetensors").is_file(), f"{label} adapter missing")
                _check(metadata["decision_labels_used_for_adapter"] == "false", f"{label} labels")
                if condition == "random_core_distillation":
                    _check(metadata["decision_core_sha256"] == random_sha256, f"{label} core")
                    _check((artifact / "head.safetensors").is_file(), f"{label} control core")
                else:
                    _check(metadata["decision_core_sha256"] == core_sha256, f"{label} core")
                    _check(
                        not (artifact / "head.safetensors").exists(),
                        f"{label} copied the Open-Jev head",
                    )
                training = record["training"]
                if condition in TRAINED:
                    gradients = training["gradient_audit"]
                    _check(
                        gradients["adapter_tensors"] > 0
                        and gradients["adapter_tensors_with_gradient"]
                        == gradients["adapter_tensors"],
                        f"{label} adapter gradients",
                    )
                    _check(gradients["core_tensors_with_gradient"] == 0, f"{label} core grads")
                    _check(gradients["backbone_tensors_with_gradient"] == 0, f"{label} LM grads")
                    _check(len(training["epoch_losses"]) == 10, f"{label} epochs")
                else:
                    _check(training is None, f"{label} untrained adapter was trained")
                for split in EVALUATION_SPLITS:
                    _check_metric_counts(
                        record["metrics"][split], counts[split], f"{label} {split}"
                    )
                for split in SPLITS:
                    match = record["decision_match"][split]
                    _check(
                        match["overall"]["example_count"] == counts[split]["total"]
                        and set(match["by_decision_type"]) == KINDS,
                        f"{label} {split} teacher-match coverage",
                    )
    _check(len(random_hashes) == len(SEEDS), "random control cores are not seed-specific")

    for name in TARGETS:
        metrics = summary["targets"][name]["metrics"]
        for condition in CONDITIONS:
            for split in EVALUATION_SPLITS:
                by_type = metrics[condition][split]["by_decision_type"]
                _check(set(by_type) == KINDS, f"aggregate {name} {condition} {split} types")
                _check(
                    all(set(by_type[kind]["accuracy"]) == {"mean", "standard_deviation"}
                        for kind in KINDS),
                    f"aggregate {name} {condition} {split} accuracy statistics",
                )

    core_check = None
    if args.verify_core:
        from decport.openjev import (
            download_openjev_package,
            load_openjev_core,
            verify_openjev_package,
        )

        package = verify_openjev_package(download_openjev_package())
        core_check = load_openjev_core(package).state_sha256()
        _check(core_check == core_sha256, "the reloaded Open-Jev core differs from the run's core")

    print(
        json.dumps(
            {
                "status": "passed",
                "seeds": SEEDS,
                "targets": sorted(TARGETS),
                "conditions_per_target": list(CONDITIONS),
                "decision_core_sha256": core_sha256,
                "mismatched_teacher_fixed_points_per_seed": EXPECTED_FIXED_POINTS,
                "decision_core_reloaded_after_run_sha256": core_check,
                "teacher_core_parity_max_abs_diff": teacher["core_parity_max_abs_diff"],
                "data_hashes_and_composition_verified": True,
                "every_decision_counted_in_every_metric": True,
                "gradients_only_in_target_adapters": True,
                "strict_label_free_boundary_verified": True,
                "same_openjev_core_for_all_targets_and_seeds": True,
                "all_recorded_numbers_finite": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _expected_counts(manifest):
    counts = {}
    for split in SPLITS:
        record = manifest["splits"][split]
        by_kind = {MANIFEST_KINDS[kind]: count for kind, count in record["decision_types"].items()}
        _check(set(by_kind) == KINDS, f"{split} manifest decision types")
        counts[split] = {"total": record["count"], "kinds": by_kind, "datasets": record["datasets"]}
    return counts


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
