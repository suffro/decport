#!/usr/bin/env python3
"""Apply the pre-registered SHIP criteria of decision 0007 and write the final verdict.

The verdict is mechanical: `decport.final_gate.evaluate_ship_gate` over every seed's results, the
JevBench public-subset (231/534) score of each target's label-free-selected learned-core adapter,
the audit, and the reproduction check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decport.final_gate import evaluate_ship_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--jevbench", required=True, help="summary.json of the JevBench run")
    parser.add_argument("--audit", required=True, help="audit.json of the run")
    parser.add_argument("--reproduction", required=True, help="reproduction_check.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    summary = _json(run / "aggregate_summary.json")
    results = [_json(run / f"seed-{seed}" / "results.json") for seed in summary["seeds"]]
    bench = _json(Path(args.jevbench))
    if not bench["complete_public_subset"]:
        raise ValueError("the JevBench run did not cover the complete public subset")
    jevbench = {}
    for target in summary["targets"]:
        seed = bench["selection"][target]["learned_core_distillation"]
        system = bench["systems"][f"{target}-learned_core_distillation-seed{seed}"]
        jevbench[target] = {
            "selected_seed": seed,
            "n_correct": system["n_correct"],
            "n_planned": system["n_planned"],
            "uniform_expected": bench["uniform_guess_expected_correct"],
        }
    audit = _json(Path(args.audit))
    reproduction = _json(Path(args.reproduction))
    gate = evaluate_ship_gate(
        results,
        jevbench,
        audit_passed=audit.get("status") == "passed",
        reproduction_passed=reproduction.get("passed") is True,
    )
    gate["jevbench_scope"] = bench["scope"]
    Path(args.output).write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"FINAL VERDICT: {gate['verdict']}")
    print(json.dumps(
        {"criteria": gate["criteria"], "targets_passing": gate["targets_passing"]}, indent=2
    ))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
