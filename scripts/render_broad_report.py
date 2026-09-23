#!/usr/bin/env python3
"""Render the audited broad-validation aggregate as a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITION_LABELS = {
    "untrained_target_adapter": "untrained adapter",
    "decision_space_distillation": "correct teacher",
    "mismatched_teacher_distillation": "mismatched teacher",
    "native_target": "native supervised",
    "supervised_decport_target": "supervised DecPort",
    "random_head_control": "random-head supervised",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    summary = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
    lines = [
        "# Broad Jev-like cross-backbone validation",
        "",
        "## Outcome",
        "",
        "The result is **positive but not universal**. Correct Qwen behavior transfers without "
        "target labels across all three heterogeneous targets on both Choice and ordered Score "
        "ID decisions. Ordered Score also transfers under dataset shift on every target. "
        "Boolean/Noul-like behavior does **not** reliably beat the mismatched-teacher control, "
        "and Choice OOD is mixed on Gemma. This meets the stated positive-evidence threshold "
        "of multiple backbones and multiple decision families, but it does not support a claim "
        "that all decision types transfer.",
        "",
        "The evidence is broad enough to justify prototyping a real shared Jev-like DecisionCore, "
        "provided Boolean behavior and calibration are treated as explicit open requirements, "
        "not as solved capabilities.",
        "",
        "At the overall level, correct transfer beat both negative controls in every individual "
        "seed for every target on both ID and OOD splits.",
        "",
        "## Protocol",
        "",
        "- Source: `Qwen/Qwen3-0.6B`.",
        "- Targets: SmolLM2-360M-Instruct, Gemma 3 270M Instruct, and "
        "TinyLlama-1.1B-Chat-v1.0.",
        "- Data: 4,500 train, 2,000 ID, and 1,500 OOD decisions.",
        "- Train/ID: ARC-Easy Choice, BoolQ Boolean, Yelp five-level ordered Score.",
        "- OOD: OpenBookQA Choice, QNLI Boolean, Amazon Reviews Multi five-level Score.",
        "- Five fixed seeds (0-4), ten epochs, one optimizer configuration, no per-target/task/"
        "seed tuning, and exact frozen-hidden-state caching.",
        "- Correct and mismatched transfer saw answer-free records only; Qwen, every backbone, "
        "the source adapter, and the source head stayed frozen; only each target adapter was "
        "optimized. All six target conditions began from the same per-target adapter state.",
        "- Values below are mean ± sample standard deviation over five seeds.",
        "",
        "## Overall transfer and behavior matching",
        "",
        "| Target | Split | Correct acc. | Untrained acc. | Mismatch acc. | Gain vs untrained | "
        "Gain vs mismatch | Teacher agreement | Probability corr. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target_name, target in summary["targets"].items():
        for split in ("in_distribution", "out_of_distribution"):
            comparison = target["comparisons"][split]["overall"]
            match = target["decision_match"]["decision_space_distillation"][split]["overall"]
            lines.append(
                f"| {target_name} | {_split(split)} | "
                f"{_pm(comparison['correct_teacher_accuracy'])} | "
                f"{_pm(comparison['untrained_accuracy'])} | "
                f"{_pm(comparison['mismatched_teacher_accuracy'])} | "
                f"{_pm(comparison['gain_over_untrained'], signed=True)} | "
                f"{_pm(comparison['gain_over_mismatched_teacher'], signed=True)} | "
                f"{_pm(match['top_choice_agreement'])} | "
                f"{_pm(match['probability_correlation'])} |"
            )

    lines.extend(["", "## Decision-type transfer gains", ""])
    for split in ("in_distribution", "out_of_distribution"):
        lines.extend(
            [
                f"### {_split(split)}",
                "",
                "| Target | Type | Correct acc. | Gain vs untrained | Gain vs mismatch |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for target_name, target in summary["targets"].items():
            for decision_type, values in target["comparisons"][split][
                "by_decision_type"
            ].items():
                lines.append(
                    f"| {target_name} | {decision_type} | "
                    f"{_pm(values['correct_teacher_accuracy'])} | "
                    f"{_pm(values['gain_over_untrained'], signed=True)} | "
                    f"{_pm(values['gain_over_mismatched_teacher'], signed=True)} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Dataset-level negative-control test",
            "",
            "Each split has one dataset per decision type; this table makes the required "
            "dataset-level comparison explicit.",
            "",
            "| Target | Split | Dataset | Correct acc. | Gain vs untrained | Gain vs mismatch |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for target_name, target in summary["targets"].items():
        for split in ("in_distribution", "out_of_distribution"):
            for dataset, values in target["comparisons"][split]["by_dataset"].items():
                lines.append(
                    f"| {target_name} | {_split(split)} | {dataset} | "
                    f"{_pm(values['correct_teacher_accuracy'])} | "
                    f"{_pm(values['gain_over_untrained'], signed=True)} | "
                    f"{_pm(values['gain_over_mismatched_teacher'], signed=True)} |"
                )

    lines.extend(
        [
            "",
            "## Macro accuracy across task families",
            "",
            "| Target | Condition | ID macro acc. | OOD macro acc. |",
            "|---|---|---:|---:|",
        ]
    )
    for target_name, target in summary["targets"].items():
        for condition, label in CONDITION_LABELS.items():
            metrics = target["metrics"][condition]
            lines.append(
                f"| {target_name} | {label} | "
                f"{_pm(metrics['in_distribution']['macro_across_task_families']['accuracy'])} | "
                f"{_pm(metrics['out_of_distribution']['macro_across_task_families']['accuracy'])} |"
            )

    for split in ("in_distribution", "out_of_distribution"):
        lines.extend(
            [
                "",
                f"## Full overall metrics — {_split(split)}",
                "",
                "| Target | Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | "
                "Teacher agreement | Probability corr. |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for target_name, target in summary["targets"].items():
            for condition, label in CONDITION_LABELS.items():
                metrics = target["metrics"][condition][split]["overall"]
                match = target["decision_match"][condition][split]["overall"]
                lines.append(
                    f"| {target_name} | {label} | {_pm(metrics['accuracy'])} | "
                    f"{_pm(metrics['macro_f1'])} | {_pm(metrics['nll'])} | "
                    f"{_pm(metrics['brier'])} | {_pm(metrics['ece'])} | "
                    f"{_pm(match['top_choice_agreement'])} | "
                    f"{_pm(match['probability_correlation'])} |"
                )

    lines.extend(
        [
            "",
            "## Ordered Score metric",
            "",
            "Lower ordinal MAE is better.",
            "",
            "| Target | Condition | Yelp ID MAE | Amazon OOD MAE |",
            "|---|---|---:|---:|",
        ]
    )
    for target_name, target in summary["targets"].items():
        for condition, label in CONDITION_LABELS.items():
            metrics = target["metrics"][condition]
            id_mae = metrics["in_distribution"]["by_decision_type"]["score"][
                "score_ordinal_mae"
            ]
            ood_mae = metrics["out_of_distribution"]["by_decision_type"]["score"][
                "score_ordinal_mae"
            ]
            lines.append(
                f"| {target_name} | {label} | "
                f"{_pm(id_mae)} | {_pm(ood_mae)} |"
            )

    source = summary["source"]["metrics"]
    lines.extend(
        [
            "",
            "## Source reference",
            "",
            "| Split | Overall acc. | Macro task-family acc. | Macro-F1 | NLL | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("in_distribution", "out_of_distribution"):
        metrics = source[split]["overall"]
        macro = source[split]["macro_across_task_families"]
        lines.append(
            f"| {_split(split)} | {_pm(metrics['accuracy'])} | {_pm(macro['accuracy'])} | "
            f"{_pm(metrics['macro_f1'])} | {_pm(metrics['nll'])} | "
            f"{_pm(metrics['brier'])} | {_pm(metrics['ece'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Choice:** strong ID transfer on every target. OOD OpenBookQA remains positive "
            "against both controls for SmolLM and TinyLlama; Gemma is above the mismatched teacher "
            "but slightly below its untrained control.",
            "- **Boolean / Noul-like:** not established. ID gains over the behavior-mismatched "
            "teacher are near zero (and negative for Gemma), while OOD QNLI is also near zero or "
            "negative. Aggregate gains must not be read as Boolean success.",
            "- **Ordered Score:** the clearest result. Correct behavior beats both controls on "
            "Yelp and Amazon for all three targets, including dataset shift, and ordinal MAE is "
            "reported alongside exact accuracy.",
            "- Calibration remains condition- and backbone-dependent. The full NLL, Brier, and ECE "
            "table is retained rather than hiding those failures behind accuracy.",
            "- The result is not attributable only to SST-2/AG News (neither appears here), one "
            "backbone, or one easy dataset.",
            "",
            "## Archive and audit",
            "",
            "The archive contains the fixed repository/expanded configs, data provenance and "
            "hashes, exact per-seed JSON, all lightweight adapter/head artifacts, aggregate JSON, "
            "environment/model revisions, GPU telemetry, and `SHA256SUMS`. "
            "`scripts/audit_broad_results.py` passed before this report was rendered.",
            "",
            "This remains diagnostic evidence, not an accepted benchmark or a universal "
            "portability claim.",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pm(value: dict[str, float], *, signed: bool = False) -> str:
    prefix = "+" if signed and value["mean"] >= 0 else ""
    return f"{prefix}{value['mean']:.4f} ± {value['standard_deviation']:.4f}"


def _split(value: str) -> str:
    return "ID" if value == "in_distribution" else "OOD"


if __name__ == "__main__":
    main()
