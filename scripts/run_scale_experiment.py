#!/usr/bin/env python3
"""Run the five-seed, two-target decision-transfer scale experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decport.backbones import GemmaBackbone, QwenBackbone, SmolLMBackbone
from decport.distillation import DistillationConfig
from decport.scale_experiment import ScaleExperimentConfig, TargetSpec, run_scale_experiment
from decport.train import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--ood", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--qwen-model")
    parser.add_argument("--smollm-model")
    parser.add_argument("--gemma-model")
    parser.add_argument("--device")
    parser.add_argument("--shared-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--cache-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--centered-logit-mse-weight", type=float)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--permutation-trials", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _load_settings(args.config)
    seeds = tuple(args.seeds or settings.get("seeds", (0, 1, 2, 3, 4)))
    learning_rate = _number(settings, "learning_rate", args.learning_rate, 1e-3)
    weight_decay = _number(settings, "weight_decay", args.weight_decay, 0.01)
    epochs = _integer(settings, "epochs", args.epochs, 5)
    config = ScaleExperimentConfig(
        train_path=args.train,
        eval_path=args.eval,
        ood_path=args.ood,
        output_dir=args.output,
        targets=(
            TargetSpec(
                "smollm",
                "smollm",
                _string(
                    settings,
                    "smollm_backbone",
                    args.smollm_model,
                    SmolLMBackbone.DEFAULT_MODEL_ID,
                ),
            ),
            TargetSpec(
                "gemma",
                "gemma",
                _string(
                    settings,
                    "gemma_backbone",
                    args.gemma_model,
                    GemmaBackbone.DEFAULT_MODEL_ID,
                ),
            ),
        ),
        seeds=seeds,
        qwen_model_id=_string(
            settings, "source_backbone", args.qwen_model, QwenBackbone.DEFAULT_MODEL_ID
        ),
        shared_size=_integer(settings, "shared_size", args.shared_size, 256),
        max_length=_integer(settings, "max_length", args.max_length, 512),
        cache_batch_size=_integer(settings, "cache_batch_size", args.cache_batch_size, 32),
        device=_string(settings, "device", args.device, "auto"),
        permutation_trials=_integer(settings, "permutation_trials", args.permutation_trials, 3),
        decision_training=TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        distillation_training=DistillationConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            temperature=_number(settings, "temperature", args.temperature, 2.0),
            centered_logit_mse_weight=_number(
                settings,
                "centered_logit_mse_weight",
                args.centered_logit_mse_weight,
                0.1,
            ),
        ),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(
        json.dumps(
            {
                "train": args.train,
                "eval": args.eval,
                "ood": args.ood,
                "output": args.output,
                "source_backbone": config.qwen_model_id,
                "targets": [
                    {"name": target.name, "model_id": target.model_id} for target in config.targets
                ],
                "seeds": list(config.seeds),
                "shared_size": config.shared_size,
                "max_length": config.max_length,
                "cache_batch_size": config.cache_batch_size,
                "device": config.device,
                "permutation_trials": config.permutation_trials,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "temperature": config.distillation_training.temperature,
                "centered_logit_mse_weight": (
                    config.distillation_training.centered_logit_mse_weight
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("starting frozen-backbone cache and scale experiment", file=sys.stderr)
    summary = run_scale_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_settings(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment config must contain a JSON object")
    return value


def _string(settings, key, override, default):
    value = override if override is not None else settings.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(settings, key, override, default):
    value = override if override is not None else settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(settings, key, override, default):
    value = override if override is not None else settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


if __name__ == "__main__":
    main()
