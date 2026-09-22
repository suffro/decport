#!/usr/bin/env python3
"""Run the multi-seed label-free decision-space distillation diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from decport.alignment import AlignmentConfig
from decport.alignment_experiment import (
    AlignmentExperimentConfig,
    aggregate_distillation_results,
    run_alignment_experiment,
)
from decport.distillation import DistillationConfig
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
    parser.add_argument("--device")
    parser.add_argument("--shared-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--centered-logit-mse-weight", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--permutation-trials", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    settings = _load_settings(args.config)
    results = []
    for seed in args.seeds:
        output_dir = str(Path(args.output) / f"seed-{seed}")
        print(f"starting seed {seed}: {output_dir}", file=sys.stderr, flush=True)
        result = run_alignment_experiment(
            _make_config(args, settings, output_dir, seed)
        )
        results.append(result)
        print(f"completed seed {seed}", file=sys.stderr, flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate_distillation_results(results)
    summary_path = Path(args.output) / "distillation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _make_config(args, settings, output_dir, seed):
    learning_rate = _number(settings, "learning_rate", args.learning_rate, 1e-3)
    weight_decay = _number(settings, "weight_decay", args.weight_decay, 0.01)
    return AlignmentExperimentConfig(
        train_path=args.train,
        eval_path=args.eval,
        ood_path=args.ood,
        output_dir=output_dir,
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
            epochs=args.epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        ),
        alignment_training=AlignmentConfig(
            epochs=args.epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        ),
        alignment_methods=("cosine_mse",),
        distillation_training=DistillationConfig(
            epochs=args.epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            temperature=args.temperature,
            centered_logit_mse_weight=args.centered_logit_mse_weight,
            seed=seed,
        ),
    )


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
