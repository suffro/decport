#!/usr/bin/env python3
"""Render Markdown result tables for an Open-Jev DecisionCore transfer run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = {"in_distribution": "ID", "out_of_distribution": "OOD"}
CONDITION_LABELS = {
    "untrained_target_adapter": "untrained adapter",
    "openjev_teacher_distillation": "Open-Jev teacher",
    "mismatched_teacher_distillation": "mismatched teacher",
    "random_core_distillation": "random frozen core",
}
KINDS = ("choice", "noul", "score")
SHOW_DEVIATION = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    root = Path(parser.parse_args().run)
    summary = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
    teacher = summary["teacher"]["metrics"]
    global SHOW_DEVIATION
    SHOW_DEVIATION = summary["seed_count"] > 1
    lines = [f"Seeds: {summary['seeds']} (mean ± sample SD when more than one seed).", ""]

    lines += _section("Overall accuracy and teacher agreement")
    lines += _table(
        ["Target", "Split", "Teacher acc.", "Correct acc.", "Untrained", "Mismatch",
         "Random core", "Agreement", "Δ vs mismatch", "Δ vs random core"]
    )
    for target, values in sorted(summary["targets"].items()):
        for split, label in SPLITS.items():
            gains = values["comparisons"][split]["overall"]
            lines.append(_row([
                target, label, _plain(teacher[split]["overall"]["accuracy"]),
                _stat(gains["correct_teacher_accuracy"]), _stat(gains["untrained_accuracy"]),
                _stat(gains["mismatched_teacher_accuracy"]), _stat(gains["random_core_accuracy"]),
                _stat(gains["correct_teacher_agreement"]),
                _stat(gains["accuracy_gain_over_mismatched_teacher"], signed=True),
                _stat(gains["accuracy_gain_over_random_core"], signed=True),
            ]))

    for split, label in SPLITS.items():
        lines += _section(f"By decision type — {label}")
        lines += _table(
            ["Target", "Type", "Teacher acc.", "Correct acc.", "Agreement", "Δ acc. vs untrained",
             "Δ acc. vs mismatch", "Δ agreement vs mismatch", "Δ acc. vs random core"]
        )
        for target, values in sorted(summary["targets"].items()):
            for kind, gains in sorted(values["comparisons"][split]["by_decision_type"].items()):
                lines.append(_row([
                    target, kind, _plain(teacher[split]["by_decision_type"][kind]["accuracy"]),
                    _stat(gains["correct_teacher_accuracy"]),
                    _stat(gains["correct_teacher_agreement"]),
                    _stat(gains["accuracy_gain_over_untrained"], signed=True),
                    _stat(gains["accuracy_gain_over_mismatched_teacher"], signed=True),
                    _stat(gains["agreement_gain_over_mismatched_teacher"], signed=True),
                    _stat(gains["accuracy_gain_over_random_core"], signed=True),
                ]))

    lines += _section("Noul behavior")
    lines += _table(["Target", "Split", "Condition", "Accuracy", "Mean P(true)",
                     "Predicted true rate", "Label true rate", "KL to teacher"])
    for split, label in SPLITS.items():
        noul = teacher[split]["by_decision_type"]["noul"]
        lines.append(_row(["teacher", label, "Open-Jev 2B", _plain(noul["accuracy"]),
                           _plain(noul["noul_mean_p_true"]),
                           _plain(noul["noul_predicted_true_rate"]),
                           _plain(noul["noul_label_true_rate"]), "—"]))
    for target, values in sorted(summary["targets"].items()):
        for split, label in SPLITS.items():
            for condition, name in CONDITION_LABELS.items():
                metrics = values["metrics"][condition][split]["by_decision_type"]["noul"]
                match = values["decision_match"][condition][split]["by_decision_type"]["noul"]
                lines.append(_row([
                    target, label, name, _stat(metrics["accuracy"]),
                    _stat(metrics["noul_mean_p_true"]), _stat(metrics["noul_predicted_true_rate"]),
                    _stat(metrics["noul_label_true_rate"]), _stat(match["kl_divergence"]),
                ]))

    lines += _section("Teacher matching by decision type")
    lines += _table(["Target", "Split", "Type", "Condition", "Top-choice agreement",
                     "KL to teacher", "Probability corr."])
    for target, values in sorted(summary["targets"].items()):
        for split, label in SPLITS.items():
            for kind in KINDS:
                for condition, name in CONDITION_LABELS.items():
                    match = values["decision_match"][condition][split]["by_decision_type"][kind]
                    lines.append(_row([
                        target, label, kind, name, _stat(match["top_choice_agreement"]),
                        _stat(match["kl_divergence"]), _stat(match["probability_correlation"]),
                    ]))

    lines += _section("Score ordinal MAE (lower is better)")
    lines += _table(["Target", "Split", "Condition", "Ordinal MAE", "Expected-value MAE"])
    for split, label in SPLITS.items():
        score = teacher[split]["by_decision_type"]["score"]
        lines.append(_row(["teacher", label, "Open-Jev 2B", _plain(score["score_ordinal_mae"]),
                           _plain(score["score_expected_value_mae"])]))
    for target, values in sorted(summary["targets"].items()):
        for split, label in SPLITS.items():
            for condition, name in CONDITION_LABELS.items():
                score = values["metrics"][condition][split]["by_decision_type"]["score"]
                lines.append(_row([target, label, name, _stat(score["score_ordinal_mae"]),
                                   _stat(score["score_expected_value_mae"])]))

    seeds = [
        json.loads((root / f"seed-{seed}" / "results.json").read_text(encoding="utf-8"))
        for seed in summary["seeds"]
    ]
    lines += _section("Seed consistency (seeds where the correct teacher wins on accuracy)")
    lines += _table(["Target", "Split", "Group", "> untrained", "> mismatch", "> random core"])
    for target in sorted(summary["targets"]):
        for split, label in SPLITS.items():
            for group in ("overall", *KINDS):
                cells = [target, label, group]
                for control in ("untrained", "mismatched_teacher", "random_core"):
                    wins = 0
                    for result in seeds:
                        comparisons = result["targets"][target]["comparisons"][split]
                        gains = (
                            comparisons["overall"] if group == "overall"
                            else comparisons["by_decision_type"][group]
                        )
                        wins += gains[f"accuracy_gain_over_{control}"] > 0
                    cells.append(f"{wins}/{len(seeds)}")
                lines.append(_row(cells))

    lines += _section("Calibration (overall)")
    lines += _table(["Target", "Split", "Condition", "NLL", "Brier", "ECE", "KL to teacher"])
    for split, label in SPLITS.items():
        overall = teacher[split]["overall"]
        lines.append(_row(["teacher", label, "Open-Jev 2B", _plain(overall["nll"]),
                           _plain(overall["brier"]), _plain(overall["ece"]), "—"]))
    for target, values in sorted(summary["targets"].items()):
        for split, label in SPLITS.items():
            for condition, name in CONDITION_LABELS.items():
                metrics = values["metrics"][condition][split]["overall"]
                match = values["decision_match"][condition][split]["overall"]
                lines.append(_row([
                    target, label, name, _stat(metrics["nll"]), _stat(metrics["brier"]),
                    _stat(metrics["ece"]), _stat(match["kl_divergence"]),
                ]))
    print("\n".join(lines))


def _section(title):
    return ["", f"### {title}", ""]


def _table(headers):
    return [_row(headers), "|" + "|".join("---" if index < 3 else "---:" for index in range(
        len(headers))) + "|"]


def _row(cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _plain(value):
    return f"{value:.4f}"


def _stat(value, signed=False):
    mean, deviation = value["mean"], value["standard_deviation"]
    text = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    return f"{text} ± {deviation:.4f}" if SHOW_DEVIATION else text


if __name__ == "__main__":
    main()
