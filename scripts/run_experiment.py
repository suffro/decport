#!/usr/bin/env python3
"""Run the four controlled DecPort v0.1 experiment conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decport.experiment import (
    ExperimentConfig,
    aggregate_convergence_results,
    run_experiment,
)
from decport.train import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training JSONL")
    parser.add_argument("--eval", required=True, help="In-distribution evaluation JSONL")
    parser.add_argument("--ood", help="Optional held-out-family evaluation JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Optional JSON hyperparameter config")
    parser.add_argument("--qwen-model")
    parser.add_argument("--smollm-model")
    parser.add_argument("--device")
    parser.add_argument("--shared-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--permutation-trials", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _load_settings(args.config)
    if args.seeds is None:
        config = _make_config(args, settings, args.output, args.seed)
        result = run_experiment(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    results = []
    for seed in args.seeds:
        seed_output = str(Path(args.output) / f"seed-{seed}")
        print(f"starting seed {seed}: {seed_output}", file=sys.stderr, flush=True)
        results.append(run_experiment(_make_config(args, settings, seed_output, seed)))
        print(f"completed seed {seed}", file=sys.stderr, flush=True)
    summary = aggregate_convergence_results(results)
    summary_path = Path(args.output) / "convergence_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _make_config(
    args: argparse.Namespace,
    settings: dict[str, object],
    output_dir: str,
    seed_override: int | None,
) -> ExperimentConfig:
    config = ExperimentConfig(
        train_path=args.train,
        eval_path=args.eval,
        ood_path=args.ood,
        output_dir=output_dir,
        qwen_model_id=_string_setting(
            settings, "source_backbone", args.qwen_model, "Qwen/Qwen3-0.6B"
        ),
        smollm_model_id=_string_setting(
            settings,
            "target_backbone",
            args.smollm_model,
            "HuggingFaceTB/SmolLM2-360M-Instruct",
        ),
        shared_size=_integer_setting(settings, "shared_size", args.shared_size, 256),
        max_length=_integer_setting(settings, "max_length", args.max_length, 512),
        device=_string_setting(settings, "device", args.device, "auto"),
        permutation_trials=_integer_setting(
            settings, "permutation_trials", args.permutation_trials, 3
        ),
        training=TrainingConfig(
            epochs=_integer_setting(settings, "epochs", args.epochs, 3),
            learning_rate=_float_setting(
                settings, "learning_rate", args.learning_rate, 1e-3
            ),
            weight_decay=_float_setting(
                settings, "weight_decay", args.weight_decay, 0.01
            ),
            seed=_integer_setting(settings, "seed", seed_override, 0),
        ),
    )
    return config


def _load_settings(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment config must contain a JSON object")
    return value


def _string_setting(
    settings: dict[str, object], key: str, override: str | None, default: str
) -> str:
    value = override if override is not None else settings.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer_setting(
    settings: dict[str, object], key: str, override: int | None, default: int
) -> int:
    value = override if override is not None else settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_setting(
    settings: dict[str, object], key: str, override: float | None, default: float
) -> float:
    value = override if override is not None else settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


if __name__ == "__main__":
    main()
