#!/usr/bin/env python3
"""Run the first broad Jev-like DecPort validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decport.backbones import QwenBackbone
from decport.broad_experiment import (
    BroadExperimentConfig,
    BroadTargetSpec,
    run_broad_experiment,
)
from decport.distillation import DistillationConfig
from decport.train import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--ood", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    training = TrainingConfig(
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    distillation = DistillationConfig(
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        temperature=settings["temperature"],
        centered_logit_mse_weight=settings["centered_logit_mse_weight"],
    )
    config = BroadExperimentConfig(
        train_path=args.train,
        eval_path=args.eval,
        ood_path=args.ood,
        output_dir=args.output,
        targets=(
            BroadTargetSpec("smollm", "smollm", settings["smollm_backbone"]),
            BroadTargetSpec("gemma", "gemma", settings["gemma_backbone"]),
            BroadTargetSpec("tinyllama", "llama", settings["tinyllama_backbone"]),
        ),
        seeds=tuple(settings["seeds"]),
        qwen_model_id=settings.get("source_backbone", QwenBackbone.DEFAULT_MODEL_ID),
        shared_size=settings["shared_size"],
        max_length=settings["max_length"],
        cache_batch_size=settings["cache_batch_size"],
        train_batch_size=settings["train_batch_size"],
        eval_batch_size=settings["eval_batch_size"],
        device=settings["device"],
        permutation_trials=settings["permutation_trials"],
        decision_training=training,
        distillation_training=distillation,
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
                "fixed_across_backbones_tasks_and_seeds": True,
                "target_conditions": [
                    "untrained_target_adapter",
                    "decision_space_distillation",
                    "mismatched_teacher_distillation",
                    "native_target",
                    "supervised_decport_target",
                    "random_head_control",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("starting broad frozen-backbone validation", file=sys.stderr)
    summary = run_broad_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
