#!/usr/bin/env python3
"""Audit archive completeness and strict transfer invariants for the broad run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = {
    "untrained_target_adapter",
    "decision_space_distillation",
    "mismatched_teacher_distillation",
    "native_target",
    "supervised_decport_target",
    "random_head_control",
}
FROZEN_SOURCE_HEAD = CONDITIONS - {"native_target", "random_head_control"}
TARGETS = {"smollm", "gemma", "tinyllama"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    summary = _json(root / "aggregate_summary.json")
    if summary["seeds"] != [0, 1, 2, 3, 4] or summary["seed_count"] != 5:
        raise RuntimeError("expected complete seeds 0-4")
    if set(summary["targets"]) != TARGETS:
        raise RuntimeError("expected all three target families")
    _assert_finite(summary)

    for seed in summary["seeds"]:
        seed_root = root / f"seed-{seed}"
        result = _json(seed_root / "results.json")
        audit = result["label_free_audit"]
        if not audit["transfer_entry_point_rejects_labeled_records"]:
            raise RuntimeError(f"seed {seed} lacks the labeled-record guard")
        if audit["distillation_examples_with_answers"] != 0:
            raise RuntimeError(f"seed {seed} contains labeled transfer records")
        if audit["ground_truth_used_for_target_distillation"]:
            raise RuntimeError(f"seed {seed} used labels in transfer")
        if audit["label_based_early_stopping_selection_weighting_or_tuning"]:
            raise RuntimeError(f"seed {seed} used label-based transfer selection")
        if not audit["teacher_outputs_detached"]:
            raise RuntimeError(f"seed {seed} teacher outputs were attached")
        source_hash = _sha256(seed_root / "source" / "head.safetensors")
        if source_hash != result["source"]["head_sha256"]:
            raise RuntimeError(f"seed {seed} source-head hash mismatch")
        for target_name in TARGETS:
            target = result["targets"][target_name]
            if target["source_head_sha256"] != source_hash:
                raise RuntimeError(f"seed {seed} {target_name} did not share the source head")
            if set(target["conditions"]) != CONDITIONS:
                raise RuntimeError(f"seed {seed} {target_name} conditions are incomplete")
            if not target["matched_adapter_initialization"]:
                raise RuntimeError(f"seed {seed} {target_name} adapter initialization mismatch")
            for condition in CONDITIONS:
                artifact = seed_root / "targets" / target_name / condition
                config = _json(artifact / "config.json")
                if condition in FROZEN_SOURCE_HEAD:
                    if config["includes_head"] or (artifact / "head.safetensors").exists():
                        raise RuntimeError(
                            f"seed {seed} {target_name} {condition} copied the source head"
                        )
                if condition in {
                    "decision_space_distillation",
                    "mismatched_teacher_distillation",
                }:
                    if config["metadata"].get("decision_labels_used_for_adapter") != "false":
                        raise RuntimeError(
                            f"seed {seed} {target_name} {condition} label audit missing"
                        )
        _assert_finite(result)

    manifest = _json(Path(args.data) / "data_manifest.json")
    expected = {
        "train": ("train.jsonl", 4500),
        "in_distribution": ("eval.jsonl", 2000),
        "out_of_distribution": ("ood.jsonl", 1500),
    }
    for split, (filename, count) in expected.items():
        record = manifest["splits"][split]
        if record["count"] != count or record["decision_types"] != {
            "boolean": count // 3 if split != "in_distribution" else 700,
            "choice": count // 3 if split != "in_distribution" else 570,
            "score": count // 3 if split != "in_distribution" else 730,
        }:
            raise RuntimeError(f"{split} data composition mismatch")
        if _sha256(Path(args.data) / filename) != record["sha256"]:
            raise RuntimeError(f"{split} checksum mismatch")
    print(
        json.dumps(
            {
                "status": "passed",
                "seeds": [0, 1, 2, 3, 4],
                "targets": sorted(TARGETS),
                "conditions_per_target": len(CONDITIONS),
                "data_hashes_and_composition_verified": True,
                "same_source_head_within_each_seed_verified": True,
                "strict_label_free_boundary_verified": True,
                "all_recorded_numbers_finite": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


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
        raise RuntimeError("result contains a non-finite number")


if __name__ == "__main__":
    main()
