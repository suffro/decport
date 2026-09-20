"""Minimal data utilities for DecPort's unified decision format."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from decport.schema import DecisionExample


def load_jsonl(path: str | Path) -> list[DecisionExample]:
    """Load validated decision examples from a UTF-8 JSON Lines file."""

    source = Path(path)
    examples: list[DecisionExample] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {source}:{line_number}")
            try:
                examples.append(DecisionExample.from_dict(value))
            except (TypeError, ValueError) as error:
                raise type(error)(f"{source}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"no decision examples found in {source}")
    return examples


def write_jsonl(examples: Iterable[DecisionExample], path: str | Path) -> None:
    """Write canonical decision records, one per line."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def shuffle_options(
    example: DecisionExample,
    rng: random.Random,
) -> DecisionExample:
    """Return an equivalent example with a freshly shuffled option order."""

    options = list(example.options)
    rng.shuffle(options)
    return DecisionExample(
        state=example.state,
        question=example.question,
        options=tuple(options),
        answer=example.answer,
    )


def convert_sst2(row: Mapping[str, Any]) -> DecisionExample:
    """Convert one GLUE/SST-2 row."""

    labels = ("negative", "positive")
    label = _integer_label(row, labels)
    return DecisionExample(
        state=_text_field(row, "sentence"),
        question="What is the sentiment of this text?",
        options=labels,
        answer=labels[label],
    )


def convert_ag_news(row: Mapping[str, Any]) -> DecisionExample:
    """Convert one AG News row."""

    labels = ("world", "sports", "business", "science and technology")
    label = _integer_label(row, labels)
    return DecisionExample(
        state=_text_field(row, "text"),
        question="Which topic best describes this news article?",
        options=labels,
        answer=labels[label],
    )


def convert_boolq(row: Mapping[str, Any]) -> DecisionExample:
    """Convert one BoolQ row; useful as a held-out Boolean task family."""

    answer = row.get("answer")
    if not isinstance(answer, bool):
        raise TypeError("BoolQ answer must be a boolean")
    return DecisionExample(
        state=_text_field(row, "passage"),
        question=_text_field(row, "question"),
        options=("yes", "no"),
        answer="yes" if answer else "no",
    )


def _text_field(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer_label(row: Mapping[str, Any], labels: tuple[str, ...]) -> int:
    value = row.get("label")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("label must be an integer")
    if value < 0 or value >= len(labels):
        raise ValueError(f"label must be between 0 and {len(labels) - 1}")
    return value
