#!/usr/bin/env python3
"""Compare a final-gate run with its independent re-execution (decision 0007).

Passes only if the two runs contain the same result files, every numeric result value agrees
within 1e-6 (wall-clock timings and output paths excluded), every non-numeric value is equal,
every safetensors artifact is byte-identical, and the pre-registered criteria A-C and E evaluate
identically on both runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decport.final_gate import evaluate_ship_gate

TOLERANCE = 1e-6
EXCLUDED_FILES = {"telemetry.json", "provenance.json", "experiment_config.json"}
PATH_KEYS = {"output_dir", "path", "output"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, second = Path(args.first), Path(args.second)

    def files(root, pattern):
        return sorted(path.relative_to(root).as_posix() for path in root.rglob(pattern))

    json_files = [name for name in files(first, "*.json") if Path(name).name not in EXCLUDED_FILES]
    tensors = files(first, "*.safetensors")
    differences: list[dict[str, object]] = []
    max_difference = 0.0
    for name in json_files:
        if not (second / name).is_file():
            differences.append({"file": name, "problem": "missing in reproduction"})
            continue
        found = _compare(_json(first / name), _json(second / name), name)
        for item in found:
            if "abs_diff" in item:
                max_difference = max(max_difference, item["abs_diff"])
        differences.extend(item for item in found if item.get("abs_diff", 1.0) > TOLERANCE)
    extra = sorted(
        set(name for name in files(second, "*.json") if Path(name).name not in EXCLUDED_FILES)
        - set(json_files)
    )
    identical_tensors = 0
    tensor_mismatches = []
    for name in tensors:
        left, right = first / name, second / name
        if right.is_file() and _sha256(left) == _sha256(right):
            identical_tensors += 1
        else:
            tensor_mismatches.append(name)
    extra_tensors = sorted(set(files(second, "*.safetensors")) - set(tensors))

    criteria = [_criteria(root) for root in (first, second)]
    passed = (
        not differences
        and not extra
        and not tensor_mismatches
        and not extra_tensors
        and criteria[0] == criteria[1]
    )
    report = {
        "kind": "independent re-execution of the identical command, code commit, config, data, "
        "and hardware",
        "passed": passed,
        "tolerance": TOLERANCE,
        "result_json_files_compared": len(json_files),
        "max_abs_numeric_difference": max_difference,
        "differences_above_tolerance": differences[:50],
        "difference_count": len(differences),
        "json_files_only_in_reproduction": extra,
        "safetensors_artifacts": len(tensors),
        "safetensors_byte_identical": identical_tensors,
        "safetensors_mismatched": tensor_mismatches,
        "safetensors_only_in_reproduction": extra_tensors,
        "criteria_a_b_c_e_identical": criteria[0] == criteria[1],
        "excluded_from_value_comparison": [
            "wall-clock seconds",
            "output paths",
            *sorted(EXCLUDED_FILES),
        ],
        "first": _times(first),
        "reproduction": _times(second),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "result_json_files_compared", "max_abs_numeric_difference", "difference_count",
        "safetensors_artifacts", "safetensors_byte_identical", "criteria_a_b_c_e_identical",
    )}, indent=2))


def _compare(left, right, where):
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        if set(left) != set(right):
            return [{"file": where, "problem": "keys differ"}]
        for key in left:
            if key in PATH_KEYS or "seconds" in key:
                continue
            found.extend(_compare(left[key], right[key], f"{where}:{key}"))
        return found
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"file": where, "problem": "lengths differ"}]
        return [item for a, b in zip(left, right) for item in _compare(a, b, where)]
    numeric = (int, float)
    if (
        isinstance(left, numeric) and isinstance(right, numeric)
        and not isinstance(left, bool) and not isinstance(right, bool)
    ):
        return [{"file": where, "abs_diff": abs(float(left) - float(right))}]
    if left != right:
        return [{"file": where, "problem": f"{left!r} != {right!r}"[:200]}]
    return []


def _criteria(root: Path) -> dict[str, object]:
    summary = _json(root / "aggregate_summary.json")
    results = [_json(root / f"seed-{seed}" / "results.json") for seed in summary["seeds"]]
    gate = evaluate_ship_gate(results, None, audit_passed=True, reproduction_passed=True)
    return {
        key: gate["criteria"][key]
        for key in (
            "A_learned_core_matters", "B_input_specific", "C_useful_under_shift",
            "E_practical_utility",
        )
    }


def _times(root: Path) -> dict[str, str | None]:
    def read(name):
        path = root / name
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    return {"start": read("run_start.txt"), "end": read("run_end.txt")}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
