#!/usr/bin/env python3
"""Prepare the initial SST-2 + AG News train/eval data and BoolQ OOD split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from decport.data import convert_ag_news, convert_boolq, convert_sst2, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument("--train-per-family", type=int, default=2000)
    parser.add_argument("--eval-per-family", type=int, default=500)
    parser.add_argument("--ood", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_per_family, args.eval_per_family, args.ood) <= 0:
        raise ValueError("all sample counts must be positive")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install DecPort with the 'train' extra") from error

    train = _take(
        load_dataset("nyu-mll/glue", "sst2", split="train"),
        args.train_per_family,
        args.seed,
    )
    train_examples = [convert_sst2(row) for row in train]
    ag_train = _take(
        load_dataset("fancyzhx/ag_news", split="train"),
        args.train_per_family,
        args.seed,
    )
    train_examples.extend(convert_ag_news(row) for row in ag_train)

    validation = _take(
        load_dataset("nyu-mll/glue", "sst2", split="validation"),
        args.eval_per_family,
        args.seed,
    )
    eval_examples = [convert_sst2(row) for row in validation]
    ag_test = _take(
        load_dataset("fancyzhx/ag_news", split="test"),
        args.eval_per_family,
        args.seed,
    )
    eval_examples.extend(convert_ag_news(row) for row in ag_test)

    boolq = _take(load_dataset("google/boolq", split="validation"), args.ood, args.seed)
    ood_examples = [convert_boolq(row) for row in boolq]

    rng = random.Random(args.seed)
    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)
    output = Path(args.output)
    write_jsonl(train_examples, output / "train.jsonl")
    write_jsonl(eval_examples, output / "eval.jsonl")
    write_jsonl(ood_examples, output / "ood_boolq.jsonl")


def _take(dataset, count: int, seed: int):
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(min(count, len(shuffled))))


if __name__ == "__main__":
    main()
