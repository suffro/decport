#!/usr/bin/env python3
"""Run the bounded label-free cross-backbone alignment diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decport.alignment import ALIGNMENT_METHODS, AlignmentConfig
from decport.alignment_experiment import (
    AlignmentExperimentConfig,
    run_alignment_experiment,
)
from decport.train import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--ood")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--qwen-model")
    parser.add_argument("--smollm-model")
    parser.add_argument("--device")
    parser.add_argument("--shared-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--decision-epochs", type=int)
    parser.add_argument("--alignment-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--alignment-learning-rate", type=float)
    parser.add_argument(
        "--alignment-methods",
        nargs="+",
        choices=ALIGNMENT_METHODS,
        default=("cosine_mse",),
    )
    parser.add_argument("--whitening-epsilon", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--permutation-trials", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _load_settings(args.config)
    seed = _integer(settings, "seed", args.seed, 0)
    learning_rate = _number(settings, "learning_rate", args.learning_rate, 1e-3)
    weight_decay = _number(settings, "weight_decay", args.weight_decay, 0.01)
    config = AlignmentExperimentConfig(
        train_path=args.train,
        eval_path=args.eval,
        ood_path=args.ood,
        output_dir=args.output,
        qwen_model_id=_string(
            settings, "source_backbone", args.qwen_model, "Qwen/Qwen3-0.6B"
        ),
        smollm_model_id=_string(
            settings,
            "target_backbone",
            args.smollm_model,
            "HuggingFaceTB/SmolLM2-360M-Instruct",
        ),
        shared_size=_integer(settings, "shared_size", args.shared_size, 256),
        max_length=_integer(settings, "max_length", args.max_length, 512),
        device=_string(settings, "device", args.device, "auto"),
        permutation_trials=_integer(
            settings, "permutation_trials", args.permutation_trials, 3
        ),
        decision_training=TrainingConfig(
            epochs=_integer(settings, "epochs", args.decision_epochs, 3),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        ),
        alignment_training=AlignmentConfig(
            epochs=(args.alignment_epochs if args.alignment_epochs is not None else 5),
            learning_rate=(
                args.alignment_learning_rate
                if args.alignment_learning_rate is not None
                else learning_rate
            ),
            weight_decay=weight_decay,
            whitening_epsilon=args.whitening_epsilon,
            ridge_alpha=args.ridge_alpha,
            seed=seed,
        ),
        alignment_methods=tuple(args.alignment_methods),
    )
    result = run_alignment_experiment(config)
    print(json.dumps(result, indent=2, sort_keys=True))


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
