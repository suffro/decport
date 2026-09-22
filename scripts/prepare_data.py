#!/usr/bin/env python3
"""Prepare the initial SST-2 + AG News train/eval data and BoolQ OOD split."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument(
        "--manifest",
        default="data_manifest.json",
        help="Manifest filename written inside the output directory",
    )
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
    _assert_disjoint(train_examples, eval_examples, ood_examples)
    output = Path(args.output)
    paths = {
        "train": output / "train.jsonl",
        "eval": output / "eval.jsonl",
        "ood_boolq": output / "ood_boolq.jsonl",
    }
    write_jsonl(train_examples, paths["train"])
    write_jsonl(eval_examples, paths["eval"])
    write_jsonl(ood_examples, paths["ood_boolq"])
    manifest = {
        "preparation": {
            "script": "scripts/prepare_data.py",
            "seed": args.seed,
            "train_per_family": args.train_per_family,
            "eval_per_family": args.eval_per_family,
            "ood": args.ood,
            "exact_record_overlap_across_splits": 0,
        },
        "splits": {
            "train": {
                "count": len(train_examples),
                "sha256": _sha256(paths["train"]),
                "families": {
                    "sst2": {
                        "count": len(train),
                        "dataset": "nyu-mll/glue",
                        "dataset_config": "sst2",
                        "split": "train",
                        "fingerprint": train._fingerprint,
                    },
                    "ag_news": {
                        "count": len(ag_train),
                        "dataset": "fancyzhx/ag_news",
                        "split": "train",
                        "fingerprint": ag_train._fingerprint,
                    },
                },
            },
            "eval": {
                "count": len(eval_examples),
                "sha256": _sha256(paths["eval"]),
                "families": {
                    "sst2": {
                        "count": len(validation),
                        "dataset": "nyu-mll/glue",
                        "dataset_config": "sst2",
                        "split": "validation",
                        "fingerprint": validation._fingerprint,
                    },
                    "ag_news": {
                        "count": len(ag_test),
                        "dataset": "fancyzhx/ag_news",
                        "split": "test",
                        "fingerprint": ag_test._fingerprint,
                    },
                },
            },
            "ood_boolq": {
                "count": len(ood_examples),
                "sha256": _sha256(paths["ood_boolq"]),
                "dataset": "google/boolq",
                "split": "validation",
                "fingerprint": boolq._fingerprint,
            },
        },
    }
    (output / args.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _take(dataset, count: int, seed: int):
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(min(count, len(shuffled))))


def _assert_disjoint(*splits) -> None:
    signatures = []
    for examples in splits:
        current = {
            hashlib.sha256(
                json.dumps(
                    {
                        "state": example.state,
                        "question": example.question,
                        "options": sorted(example.options),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            for example in examples
        }
        if len(current) != len(examples):
            raise ValueError("a prepared split contains duplicate decision records")
        signatures.append(current)
    for left_index, left in enumerate(signatures):
        for right in signatures[left_index + 1 :]:
            if left.intersection(right):
                raise ValueError("prepared train/eval/OOD splits overlap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
