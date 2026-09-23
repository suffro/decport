#!/usr/bin/env python3
"""Render Markdown tables for a final-gate run (decision 0007)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decport.final_gate import CONDITIONS

LABELS = {
    "learned_core_distillation": "A learned core",
    "random_core_distillation": "B random core",
    "mismatched_teacher_distillation": "C mismatched",
    "untrained_adapter": "D untrained",
    "target_specific_distillation": "E target-specific",
}
CONTROL_NAMES = {
    "random_core": "B random core",
    "mismatched_teacher": "C mismatched",
    "untrained": "D untrained",
    "target_specific": "E target-specific",
}
SPLITS = {"in_distribution": "ID", "out_of_distribution": "OOD"}
KINDS = ("choice", "noul", "score")
TARGET_NAMES = {"smollm": "SmolLM2-360M", "gemma": "Gemma 3 270M", "tinyllama": "TinyLlama-1.1B"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--jevbench", help="summary.json of the JevBench run")
    parser.add_argument("--verdict", help="verdict.json from decide_final_gate.py")
    parser.add_argument("--output", help="write the tables here instead of stdout")
    args = parser.parse_args()
    run = Path(args.run)
    summary = _json(run / "aggregate_summary.json")
    source = summary["source_system"]
    lines: list[str] = []
    add = lines.append

    add("## Source system")
    add("")
    add("| System | Split | Overall | Choice | Noul | Score | NLL | ECE |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for label, metrics in (
        ("Open-Jev repr. + learned core", source["metrics"]),
        ("Open-Jev 2B (own linear head)", source["openjev_head_reference_metrics"]),
    ):
        for split, name in SPLITS.items():
            value = metrics[split]
            kinds = value["by_decision_type"]
            add(
                f"| {label} | {name} | {value['overall']['accuracy']:.4f} | "
                + " | ".join(f"{kinds[kind]['accuracy']:.4f}" for kind in KINDS)
                + f" | {value['overall']['nll']:.4f} | {value['overall']['ece']:.4f} |"
            )
    calibration = source["metrics"]["source_core_calibration"]["overall"]
    add("")
    add(
        f"Learned-core temperature {source['temperature']:.4f}; source calibration-split "
        f"accuracy {calibration['accuracy']:.4f}, NLL {calibration['nll']:.4f}."
    )
    add("")

    add("## Overall accuracy (mean ± SD over seeds)")
    add("")
    add("| Target | Split | " + " | ".join(LABELS[c] for c in _order()) + " |")
    add("|---|---|" + "---:|" * len(CONDITIONS))
    for target, entry in summary["targets"].items():
        for split, name in SPLITS.items():
            cells = [_ms(entry["metrics"][c][split]["overall"]["accuracy"]) for c in _order()]
            add(f"| {TARGET_NAMES.get(target, target)} | {name} | " + " | ".join(cells) + " |")
    add("")

    add("## Paired per-seed differences: A learned core − control (overall accuracy)")
    add("")
    add("| Target | Split | Control | Mean ± SD | Seeds > 0 | Per seed |")
    add("|---|---|---|---:|---:|---|")
    for target, entry in summary["targets"].items():
        for split, name in SPLITS.items():
            for control, label in CONTROL_NAMES.items():
                value = entry["paired_accuracy_differences"][split][control]
                seeds = ", ".join(f"{item:+.4f}" for item in value["per_seed"])
                add(
                    f"| {TARGET_NAMES.get(target, target)} | {name} | {label} | "
                    f"{value['mean']:+.4f} ± {value['standard_deviation']:.4f} | "
                    f"{value['positive_seeds']}/{value['seeds']} | {seeds} |"
                )
    add("")

    add("## Accuracy by decision type (mean ± SD)")
    add("")
    add("| Target | Split | Type | Source | " + " | ".join(LABELS[c] for c in _order()) + " |")
    add("|---|---|---|---:|" + "---:|" * len(CONDITIONS))
    for target, entry in summary["targets"].items():
        for split, name in SPLITS.items():
            for kind in KINDS:
                teacher = source["metrics"][split]["by_decision_type"][kind]["accuracy"]
                cells = [
                    _ms(entry["metrics"][c][split]["by_decision_type"][kind]["accuracy"])
                    for c in _order()
                ]
                add(
                    f"| {TARGET_NAMES.get(target, target)} | {name} | {kind} | {teacher:.4f} | "
                    + " | ".join(cells) + " |"
                )
    add("")

    add("## Calibration and fidelity to the source system (overall, mean ± SD)")
    add("")
    add("| Target | Split | Condition | Macro-F1 | NLL | Brier | ECE | Agreement | KL |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for target, entry in summary["targets"].items():
        for split, name in SPLITS.items():
            for condition in _order():
                metrics = entry["metrics"][condition][split]["overall"]
                match = entry["decision_match"][condition][split]["overall"]
                add(
                    f"| {TARGET_NAMES.get(target, target)} | {name} | {LABELS[condition]} | "
                    f"{_ms(metrics['macro_f1'])} | {_ms(metrics['nll'])} | "
                    f"{_ms(metrics['brier'])} | {_ms(metrics['ece'])} | "
                    f"{_ms(match['top_choice_agreement'])} | {_ms(match['kl_divergence'])} |"
                )
    add("")

    add("## Noul and Score detail (mean over seeds)")
    add("")
    add("| Target | Split | Condition | Noul P(true) | Noul predicted-true rate | "
        "Noul train KL | Score ordinal MAE |")
    add("|---|---|---|---:|---:|---:|---:|")
    for target, entry in summary["targets"].items():
        for split, name in SPLITS.items():
            for condition in _order():
                noul = entry["metrics"][condition][split]["by_decision_type"]["noul"]
                score = entry["metrics"][condition][split]["by_decision_type"]["score"]
                train_kl = entry["decision_match"][condition]["train"]["by_decision_type"][
                    "noul"]["kl_divergence"]["mean"]
                add(
                    f"| {TARGET_NAMES.get(target, target)} | {name} | {LABELS[condition]} | "
                    f"{noul['noul_mean_p_true']['mean']:.4f} | "
                    f"{noul['noul_predicted_true_rate']['mean']:.4f} | {train_kl:.4f} | "
                    f"{score['score_ordinal_mae']['mean']:.4f} |"
                )
    add("")

    add("## Cost per target")
    add("")
    add("| Target | Condition | Trainable params | Artifact bytes | Train s | "
        "Inference s (ID+OOD, cached states) |")
    add("|---|---|---:|---:|---:|---:|")
    for target, entry in summary["targets"].items():
        for condition in _order():
            cost = entry["cost"][condition]
            inference = sum(
                cost["inference_seconds_from_cached_states"][split]["mean"] for split in SPLITS
            )
            add(
                f"| {TARGET_NAMES.get(target, target)} | {LABELS[condition]} | "
                f"{cost['trainable_parameters']['mean']:,.0f} | "
                f"{cost['artifact_bytes']['mean']:,.0f} | "
                f"{cost['training_seconds']['mean']:.1f} | {inference:.2f} |"
            )
    parameters = next(iter(summary["targets"].values()))["parameters"]
    add("")
    add(
        f"Shared learned core (stored once, not per target): "
        f"{parameters['shared_core_not_per_target']:,} parameters."
    )
    add("")

    if args.jevbench:
        bench = _json(Path(args.jevbench))
        add("## JevBench public subset (231/534)")
        add("")
        add(bench["scope"])
        add("")
        add("| System | Seed | Correct / 231 | Choice | Noul | Score | Brier | ECE | "
            "Agreement with source |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, system in bench["systems"].items():
            kinds = system["by_question_type"]
            seed = name.rsplit("-seed", 1)[1] if "-seed" in name else "—"
            add(
                f"| {name.rsplit('-seed', 1)[0]} | {seed} | {system['n_correct']} | "
                + " | ".join(
                    f"{kinds[kind][0]}/{kinds[kind][1]}" if kind in kinds else "—"
                    for kind in KINDS
                )
                + f" | {system['brier_mean']:.4f} | {system['ece_top_label']:.4f} | "
                f"{system.get('source_system_top_choice_agreement', '—')} |"
            )
        add("")
        add(f"Uniform-guess expectation: {bench['uniform_guess_expected_correct']:.2f} / 231.")
        add("")

    if args.verdict:
        verdict = _json(Path(args.verdict))
        add("## Pre-registered criteria")
        add("")
        add("| Criterion | Passing targets | Result |")
        add("|---|---|---|")
        for key, passed in verdict["criteria"].items():
            targets = ", ".join(verdict["targets_passing"][key]) or "none"
            add(f"| {key} | {targets} | {'pass' if passed else 'FAIL'} |")
        add("")
        add(f"Audit passed: {verdict['audit_passed']}. "
            f"Reproduction passed: {verdict['reproduction_passed']}.")
        add("")
        add(f"**FINAL VERDICT: {verdict['verdict']}**")

    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


def _order():
    return (
        "learned_core_distillation",
        "random_core_distillation",
        "mismatched_teacher_distillation",
        "untrained_adapter",
        "target_specific_distillation",
    )


def _ms(value) -> str:
    return f"{value['mean']:.4f} ± {value['standard_deviation']:.4f}"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
