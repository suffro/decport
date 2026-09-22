#!/usr/bin/env python3
"""Audit completeness and label-free invariants of a scale experiment archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = {
    "unaligned_target",
    "decision_space_distillation",
    "permuted_teacher_distillation",
    "native_target",
    "supervised_transfer_target",
    "random_head_control",
}
FROZEN_HEAD_CONDITIONS = {
    "unaligned_target",
    "decision_space_distillation",
    "permuted_teacher_distillation",
    "supervised_transfer_target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run)
    summary = _json(root / "aggregate_summary.json")
    seeds = summary["seeds"]
    if seeds != [0, 1, 2, 3, 4] or summary["seed_count"] != 5:
        raise RuntimeError("expected complete seeds 0-4")
    _assert_finite(summary)

    for seed in seeds:
        seed_root = root / f"seed-{seed}"
        result = _json(seed_root / "results.json")
        audit = result["label_free_audit"]
        if audit["distillation_examples_with_answers"] != 0:
            raise RuntimeError(f"seed {seed} contains labeled distillation examples")
        if audit["ground_truth_used_for_target_distillation"]:
            raise RuntimeError(f"seed {seed} used target labels for distillation")
        if audit["label_based_early_stopping_or_selection"]:
            raise RuntimeError(f"seed {seed} used label-based selection")
        if not audit["teacher_outputs_detached"]:
            raise RuntimeError(f"seed {seed} teacher outputs were not detached")
        source_head = seed_root / "source" / "head.safetensors"
        source_hash = _sha256(source_head)
        if source_hash != result["source"]["head_sha256"]:
            raise RuntimeError(f"seed {seed} source-head checksum mismatch")
        for target_name in ("smollm", "gemma"):
            target = result["targets"][target_name]
            if target["source_head_sha256"] != source_hash:
                raise RuntimeError(f"seed {seed} {target_name} did not use shared head")
            if set(target["conditions"]) != CONDITIONS:
                raise RuntimeError(f"seed {seed} {target_name} conditions are incomplete")
            if not target["matched_adapter_initialization"]:
                raise RuntimeError(f"seed {seed} {target_name} initialization was unmatched")
            for condition in CONDITIONS:
                artifact = seed_root / "targets" / target_name / condition
                artifact_config = _json(artifact / "config.json")
                if condition in FROZEN_HEAD_CONDITIONS:
                    if (artifact / "head.safetensors").exists():
                        raise RuntimeError(
                            f"seed {seed} {target_name} {condition} copied a source head"
                        )
                    if artifact_config["includes_head"]:
                        raise RuntimeError(
                            f"seed {seed} {target_name} {condition} claims an included head"
                        )
                if (
                    condition
                    in {
                        "decision_space_distillation",
                        "permuted_teacher_distillation",
                    }
                    and artifact_config["metadata"].get("decision_labels_used_for_adapter")
                    != "false"
                ):
                    raise RuntimeError(f"seed {seed} {target_name} {condition} label audit missing")
        _assert_finite(result)

    manifest = _json(Path(args.data) / "data_manifest.json")
    for split, filename in (
        ("train", "train.jsonl"),
        ("eval", "eval.jsonl"),
        ("ood_boolq", "ood_boolq.jsonl"),
    ):
        actual = _sha256(Path(args.data) / filename)
        if actual != manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"{split} data checksum mismatch")
    print(
        json.dumps(
            {
                "status": "passed",
                "seeds": seeds,
                "targets": ["smollm", "gemma"],
                "conditions_per_target": len(CONDITIONS),
                "data_hashes_verified": True,
                "same_source_head_verified_within_each_seed": True,
                "label_free_invariants_verified": True,
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
