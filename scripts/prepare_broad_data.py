#!/usr/bin/env python3
"""Prepare the broad Choice/Boolean/Score DecPort validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from decport.data import (
    convert_arc,
    convert_boolq,
    convert_qnli,
    convert_review_rating,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import datasets
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install DecPort with the 'train' extra") from error

    specs = {
        "arc_easy_train": _take(
            load_dataset("allenai/ai2_arc", "ARC-Easy", split="train"), 1500, args.seed
        ),
        "arc_easy_eval": _take(
            load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation"), 570, args.seed
        ),
        "boolq_train": _take(load_dataset("google/boolq", split="train"), 1500, args.seed),
        "boolq_eval": _take(load_dataset("google/boolq", split="validation"), 700, args.seed),
        "yelp_train": _take(load_dataset("Yelp/yelp_review_full", split="train"), 1500, args.seed),
        "yelp_eval": _take(load_dataset("Yelp/yelp_review_full", split="test"), 730, args.seed),
        "openbookqa_ood": _take(
            load_dataset("allenai/openbookqa", "main", split="validation"), 500, args.seed
        ),
        "qnli_ood": _take(load_dataset("nyu-mll/glue", "qnli", split="validation"), 500, args.seed),
        "amazon_ood": _take(
            load_dataset("SetFit/amazon_reviews_multi_en", split="test"), 500, args.seed
        ),
    }
    train = [convert_arc(row) for row in specs["arc_easy_train"]]
    train.extend(convert_boolq(row) for row in specs["boolq_train"])
    train.extend(
        convert_review_rating(row, dataset="yelp_review_full") for row in specs["yelp_train"]
    )
    evaluation = [convert_arc(row) for row in specs["arc_easy_eval"]]
    evaluation.extend(convert_boolq(row) for row in specs["boolq_eval"])
    evaluation.extend(
        convert_review_rating(row, dataset="yelp_review_full") for row in specs["yelp_eval"]
    )
    ood = [convert_arc(row, dataset="openbookqa") for row in specs["openbookqa_ood"]]
    ood.extend(convert_qnli(row) for row in specs["qnli_ood"])
    ood.extend(
        convert_review_rating(row, dataset="amazon_reviews_multi") for row in specs["amazon_ood"]
    )

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(evaluation)
    rng.shuffle(ood)
    _assert_complete_metadata(train, evaluation, ood)
    _assert_disjoint(train, evaluation, ood)

    output = Path(args.output)
    paths = {
        "train": output / "train.jsonl",
        "in_distribution": output / "eval.jsonl",
        "out_of_distribution": output / "ood.jsonl",
    }
    for split, examples in (
        ("train", train),
        ("in_distribution", evaluation),
        ("out_of_distribution", ood),
    ):
        write_jsonl(examples, paths[split])

    sources = {
        "arc_easy_train": ("allenai/ai2_arc", "ARC-Easy", "train"),
        "arc_easy_eval": ("allenai/ai2_arc", "ARC-Easy", "validation"),
        "boolq_train": ("google/boolq", None, "train"),
        "boolq_eval": ("google/boolq", None, "validation"),
        "yelp_train": ("Yelp/yelp_review_full", None, "train"),
        "yelp_eval": ("Yelp/yelp_review_full", None, "test"),
        "openbookqa_ood": ("allenai/openbookqa", "main", "validation"),
        "qnli_ood": ("nyu-mll/glue", "qnli", "validation"),
        "amazon_ood": ("SetFit/amazon_reviews_multi_en", None, "test"),
    }
    split_examples = {
        "train": train,
        "in_distribution": evaluation,
        "out_of_distribution": ood,
    }
    manifest = {
        "preparation": {
            "script": "scripts/prepare_broad_data.py",
            "seed": args.seed,
            "datasets_version": datasets.__version__,
            "exact_record_overlap_across_splits": 0,
        },
        "sources": {
            name: {
                "dataset": source[0],
                "dataset_config": source[1],
                "split": source[2],
                "count": len(specs[name]),
                "fingerprint": specs[name]._fingerprint,
            }
            for name, source in sources.items()
        },
        "splits": {
            split: {
                "file": path.name,
                "count": len(split_examples[split]),
                "sha256": _sha256(path),
                "datasets": _counts(split_examples[split], "dataset"),
                "task_families": _counts(split_examples[split], "task_family"),
                "decision_types": _counts(split_examples[split], "decision_type"),
            }
            for split, path in paths.items()
        },
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _take(dataset, count: int, seed: int):
    if len(dataset) < count:
        raise ValueError(f"requested {count} records from a split with {len(dataset)}")
    return dataset.shuffle(seed=seed).select(range(count))


def _counts(examples, field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for example in examples:
        key = getattr(example, field)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _assert_complete_metadata(*splits) -> None:
    if any(
        not example.dataset or not example.task_family or not example.decision_type
        for examples in splits
        for example in examples
    ):
        raise ValueError("every broad-validation record requires complete task metadata")


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
    for index, left in enumerate(signatures):
        for right in signatures[index + 1 :]:
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
