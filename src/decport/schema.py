"""Validated, backbone-independent decision records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class ChoiceResult(TypedDict):
    """JSON-compatible result returned by :meth:`DecPort.choice`."""

    choice: str
    probabilities: dict[str, float]
    scores: dict[str, float]


class BooleanResult(TypedDict):
    value: bool
    probabilities: dict[str, float]


class ScoreResult(TypedDict):
    score: float
    level: str
    probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class DecisionExample:
    """Unified representation of one dynamic-option decision.

    ``answer`` is omitted for inference and contains the exact correct option for
    supervised examples.
    """

    state: str
    question: str
    options: tuple[str, ...]
    answer: str | None = None
    dataset: str | None = None
    task_family: str | None = None
    decision_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, str):
            raise TypeError("state must be a string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")

        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if len(options) < 2:
            raise ValueError("a decision must contain at least two options")
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise ValueError("every option must be a non-empty string")
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        if self.answer is not None and self.answer not in options:
            raise ValueError("answer must exactly match one of the options")
        for name in ("dataset", "task_family", "decision_type"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DecisionExample:
        """Parse the canonical JSON-compatible dataset shape."""

        required = {"state", "question", "options"}
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        raw_options = value["options"]
        if not isinstance(raw_options, (list, tuple)):
            raise TypeError("options must be a list or tuple")

        state = value["state"]
        question = value["question"]
        answer = value.get("answer")
        dataset = value.get("dataset")
        task_family = value.get("task_family")
        decision_type = value.get("decision_type")
        if not isinstance(state, str):
            raise TypeError("state must be a string")
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if answer is not None and not isinstance(answer, str):
            raise TypeError("answer must be a string when provided")
        for name, metadata in (
            ("dataset", dataset),
            ("task_family", task_family),
            ("decision_type", decision_type),
        ):
            if metadata is not None and not isinstance(metadata, str):
                raise TypeError(f"{name} must be a string when provided")

        return cls(
            state=state,
            question=question,
            options=tuple(raw_options),  # type: ignore[arg-type]
            answer=answer,
            dataset=dataset,
            task_family=task_family,
            decision_type=decision_type,
        )

    @property
    def answer_index(self) -> int:
        """Return the supervised label index."""

        if self.answer is None:
            raise ValueError("an inference example has no answer index")
        return self.options.index(self.answer)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "state": self.state,
            "question": self.question,
            "options": list(self.options),
        }
        if self.answer is not None:
            result["answer"] = self.answer
        if self.dataset is not None:
            result["dataset"] = self.dataset
        if self.task_family is not None:
            result["task_family"] = self.task_family
        if self.decision_type is not None:
            result["decision_type"] = self.decision_type
        return result
