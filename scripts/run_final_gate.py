#!/usr/bin/env python3
"""Run the DecPort final ship gate (decision 0007)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decport.broad_experiment import BroadTargetSpec
from decport.distillation import DistillationConfig
from decport.final_gate import CONDITIONS, FinalGateConfig, SourceCoreConfig, run_final_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="the jev-broad-v0.1 transfer data directory")
    parser.add_argument("--source-data", required=True, help="the source-core data directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--package-dir", help="verified local Open-Jev package directory")
    return parser.parse_args()


def load_config(path: str, args: argparse.Namespace) -> tuple[FinalGateConfig, dict]:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    data, source = Path(args.data), Path(args.source_data)
    source_core = settings["source_core"]
    config = FinalGateConfig(
        train_path=str(data / "train.jsonl"),
        eval_path=str(data / "eval.jsonl"),
        ood_path=str(data / "ood.jsonl"),
        source_train_path=str(source / "source_train.jsonl"),
        source_calibration_path=str(source / "source_calibration.jsonl"),
        output_dir=args.output,
        targets=tuple(
            BroadTargetSpec(name, target["backbone_type"], target["model_id"])
            for name, target in settings["targets"].items()
        ),
        seeds=tuple(settings["seeds"]),
        package_dir=args.package_dir,
        package_manifest_sha256=settings["openjev_package_manifest_sha256"],
        adapter_rank=settings["adapter_rank"],
        core_hidden_sizes=tuple(settings["core_hidden_sizes"]),
        baseline_hidden_sizes=tuple(settings["target_specific_hidden_sizes"]),
        teacher_max_length=settings["teacher_max_length"],
        teacher_candidate_batch_size=settings["teacher_candidate_batch_size"],
        target_max_length=settings["target_max_length"],
        cache_batch_size=settings["cache_batch_size"],
        train_batch_size=settings["train_batch_size"],
        eval_batch_size=settings["eval_batch_size"],
        device=settings["device"],
        source_core=SourceCoreConfig(
            seed=source_core["seed"],
            epochs=source_core["epochs"],
            batch_size=source_core["batch_size"],
            learning_rate=source_core["learning_rate"],
            weight_decay=source_core["weight_decay"],
            max_grad_norm=source_core["max_grad_norm"],
            temperature_grid=tuple(source_core["temperature_grid"]),
        ),
        distillation=DistillationConfig(
            epochs=settings["epochs"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            max_grad_norm=settings["max_grad_norm"],
            temperature=settings["distillation_temperature"],
            centered_logit_mse_weight=settings["centered_logit_mse_weight"],
        ),
        limit_per_type=settings.get("limit_per_type"),
    )
    return config, settings


def main() -> None:
    args = parse_args()
    config, settings = load_config(args.config, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(
        json.dumps(
            {
                **settings,
                "data": args.data,
                "source_data": args.source_data,
                "output": args.output,
                "package_dir": args.package_dir,
                "target_conditions": list(CONDITIONS),
                "fixed_across_backbones_types_and_seeds": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("starting the DecPort final ship gate", file=sys.stderr)
    summary = run_final_gate(config)
    print(json.dumps({"status": summary["status"], "seeds": summary["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
