#!/usr/bin/env python3
"""Prepare the source-only labeled splits for the final-gate DecisionCore (decision 0007).

The learned core must never be trained on the transfer train split, or its outputs there would carry
memorized labels into "label-free" target training. These records come from the same three ID
datasets as `data/jev-broad-v0.1`, and exact overlap with its train/ID/OOD splits is rejected:

- Choice: ARC-Easy `test` (the broad data uses only `train` and `validation`), from the start of
  its seed-0 shuffle;
- Noul: BoolQ `train`, from record 1,500 of the seed-0 shuffle whose first 1,500 are the transfer
  train;
- Score: Yelp `train`, from record 1,500 of the same seed-0 shuffle.

Records that exactly duplicate a transfer record (or an already selected one) are skipped, and
their indices are recorded. The first 1,500 selected per type become `source_train.jsonl`; the next
300 per type become `source_calibration.jsonl`, used only to fit the core's temperature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from decport.data import convert_arc, convert_boolq, convert_review_rating, load_jsonl, write_jsonl

TRAIN_PER_TYPE = 1500
CALIBRATION_PER_TYPE = 300
BROAD_TRAIN_PER_DATASET = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--broad-data", required=True, help="the existing jev-broad-v0.1 directory")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import datasets
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install DecPort with the 'train' extra") from error

    total = TRAIN_PER_TYPE + CALIBRATION_PER_TYPE
    broad = Path(args.broad_data)
    transfer = [
        example
        for name in ("train.jsonl", "eval.jsonl", "ood.jsonl")
        for example in load_jsonl(broad / name)
    ]
    # Datasets contain a few exact duplicates of transfer records; such records are skipped, and
    # selection continues deterministically through the same shuffled pool.
    taken = {_signature(example) for example in transfer}
    pools = {
        "arc_easy_test": (
            load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").shuffle(seed=args.seed),
            ("allenai/ai2_arc", "ARC-Easy", "test"),
            0,
            convert_arc,
        ),
        "boolq_train_unused": (
            load_dataset("google/boolq", split="train").shuffle(seed=args.seed),
            ("google/boolq", None, "train"),
            BROAD_TRAIN_PER_DATASET,
            convert_boolq,
        ),
        "yelp_train_unused": (
            load_dataset("Yelp/yelp_review_full", split="train").shuffle(seed=args.seed),
            ("Yelp/yelp_review_full", None, "train"),
            BROAD_TRAIN_PER_DATASET,
            lambda row: convert_review_rating(row, dataset="yelp_review_full"),
        ),
    }
    converted, selection = {}, {}
    for name, (dataset, _, start, convert) in pools.items():
        rows, skipped, index = [], [], start
        while len(rows) < total:
            if index >= len(dataset):
                raise ValueError(f"{name} has too few unused records")
            example = convert(dataset[index])
            signature = _signature(example)
            if signature in taken:
                skipped.append(index)
            else:
                taken.add(signature)
                rows.append(example)
            index += 1
        converted[name] = rows
        selection[name] = {
            "shuffled_index_range": [start, index],
            "skipped_duplicate_indices": skipped,
            "fingerprint": dataset._fingerprint,
        }
    source_train = [row for rows in converted.values() for row in rows[:TRAIN_PER_TYPE]]
    calibration = [row for rows in converted.values() for row in rows[TRAIN_PER_TYPE:]]
    rng = random.Random(args.seed)
    rng.shuffle(source_train)
    rng.shuffle(calibration)

    source_signatures = {_signature(example) for example in source_train + calibration}
    if len(source_signatures) != len(source_train) + len(calibration):
        raise ValueError("source-core records contain duplicates")
    if source_signatures & {_signature(example) for example in transfer}:
        raise ValueError("source-core records overlap the transfer splits")

    output = Path(args.output)
    paths = {
        "source_core_train": output / "source_train.jsonl",
        "source_core_calibration": output / "source_calibration.jsonl",
    }
    write_jsonl(source_train, paths["source_core_train"])
    write_jsonl(calibration, paths["source_core_calibration"])
    split_examples = {"source_core_train": source_train, "source_core_calibration": calibration}
    manifest = {
        "preparation": {
            "script": "scripts/prepare_source_core_data.py",
            "seed": args.seed,
            "datasets_version": datasets.__version__,
            "exact_record_overlap_with_transfer_splits": 0,
            "transfer_data_manifest_sha256": _sha256(broad / "data_manifest.json"),
            "purpose": "source-side labeled DecisionCore training and calibration only",
        },
        "sources": {
            name: {
                "dataset": source[0],
                "dataset_config": source[1],
                "split": source[2],
                "count": total,
                **selection[name],
            }
            for name, (_, source, _, _) in pools.items()
        },
        "splits": {
            split: {
                "file": path.name,
                "count": len(split_examples[split]),
                "sha256": _sha256(path),
                "datasets": _counts(split_examples[split], "dataset"),
                "decision_types": _counts(split_examples[split], "decision_type"),
            }
            for split, path in paths.items()
        },
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))


def _signature(example) -> str:
    return hashlib.sha256(
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


def _counts(examples, field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for example in examples:
        key = getattr(example, field)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
