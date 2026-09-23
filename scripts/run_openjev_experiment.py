#!/usr/bin/env python3
"""Run the Open-Jev 2B DecisionCore cross-backbone transfer experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decport.broad_experiment import BroadTargetSpec
from decport.distillation import DistillationConfig
from decport.openjev_experiment import CONDITIONS, OpenJevTransferConfig, run_openjev_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--ood", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--package-dir",
        help="verified local Open-Jev package directory; downloaded from the pinned Hub revision "
        "when omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = OpenJevTransferConfig(
        train_path=args.train,
        eval_path=args.eval,
        ood_path=args.ood,
        output_dir=args.output,
        targets=tuple(
            BroadTargetSpec(name, target["backbone_type"], target["model_id"])
            for name, target in settings["targets"].items()
        ),
        seeds=tuple(settings["seeds"]),
        package_dir=args.package_dir,
        package_manifest_sha256=settings["openjev_package_manifest_sha256"],
        adapter_hidden_size=settings["adapter_hidden_size"],
        teacher_max_length=settings["teacher_max_length"],
        teacher_candidate_batch_size=settings["teacher_candidate_batch_size"],
        target_max_length=settings["target_max_length"],
        cache_batch_size=settings["cache_batch_size"],
        train_batch_size=settings["train_batch_size"],
        eval_batch_size=settings["eval_batch_size"],
        device=settings["device"],
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(
        json.dumps(
            {
                **settings,
                "train": args.train,
                "eval": args.eval,
                "ood": args.ood,
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
    print("starting Open-Jev DecisionCore transfer", file=sys.stderr)
    summary = run_openjev_experiment(config)
    print(json.dumps({"status": summary["status"], "seeds": summary["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
