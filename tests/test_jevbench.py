import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from decport.backbones import PromptTooLongError
from decport.jevbench import (
    CONTEXT_LIMIT_STATUS,
    DecPortJevBenchAdapter,
    jevbench_probabilities,
    typed_decision_from_task,
)


@dataclass
class StubResult:
    """Field-compatible stand-in for JevBench's DecisionResult."""

    adapter: str
    ok: bool
    probs: dict | None = None
    probs_source: str = "unknown"
    model: str = ""
    status: int | None = None
    error: str | None = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    raw: Any = None
    request_body: Any = None


def build_question(task):
    """Same contract as JevBench's adapters.base.build_question."""

    question = {"type": task.question["type"], "instructions": task.question["instructions"]}
    if task.question.get("criteria") is not None:
        question["criteria"] = task.question["criteria"]
    return question


UPSTREAM = SimpleNamespace(
    base=SimpleNamespace(DecisionResult=StubResult, build_question=build_question)
)


def task(kind: str):
    if kind == "noul":
        return SimpleNamespace(
            id="n1", family="policy", state={"refund": 12}, labels=["no", "yes"], expected="yes",
            question={"type": "noul", "instructions": "Allowed?", "criteria": None},
        )
    if kind == "choice":
        return SimpleNamespace(
            id="c1", family="routing", state="Charged twice.", labels=["billing", "tech"],
            expected="billing",
            question={"type": "choice", "instructions": "Route?",
                      "criteria": {"billing": "Charges", "tech": "Bugs"}},
        )
    return SimpleNamespace(
        id="s1", family="ordinal", state="Outage.", labels=["0", "1", "2"], expected=2,
        question={"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
    )


def test_tasks_convert_to_typed_decisions_without_reading_gold() -> None:
    for kind in ("noul", "choice", "score"):
        decision = typed_decision_from_task(task(kind), build_question)

        assert decision.kind == kind
        assert decision.answer is None
        assert decision.decision_id == task(kind).id
        assert decision.question() == build_question(task(kind))
        assert "expected" not in str(decision.question())


def test_predictions_map_to_the_exact_jevbench_label_set() -> None:
    noul = typed_decision_from_task(task("noul"), build_question)
    choice = typed_decision_from_task(task("choice"), build_question)
    score = typed_decision_from_task(task("score"), build_question)

    assert jevbench_probabilities(noul, [0.2, 0.8], ["no", "yes"]) == {"no": 0.2, "yes": 0.8}
    assert jevbench_probabilities(choice, [0.9, 0.1], ["billing", "tech"]) == {
        "billing": 0.9,
        "tech": 0.1,
    }
    assert set(jevbench_probabilities(score, [0.1, 0.2, 0.7], ["0", "1", "2"])) == {"0", "1", "2"}
    with pytest.raises(ValueError, match="label set"):
        jevbench_probabilities(choice, [0.5, 0.5], ["billing", "sales"])
    with pytest.raises(ValueError, match="probability count"):
        jevbench_probabilities(score, [0.5, 0.5], ["0", "1", "2"])


def test_adapter_returns_native_distributions_and_context_limit_refusals() -> None:
    adapter = DecPortJevBenchAdapter(
        lambda decision: [0.25, 0.75] if decision.kind != "score" else [0.1, 0.2, 0.7],
        name="decport_test",
        model="stub",
        upstream=UPSTREAM,
    )

    result = adapter.run(task("noul"))

    assert result.ok and result.probs == {"no": 0.25, "yes": 0.75}
    assert result.probs_source == "native"
    assert "expected" not in str(result.request_body)
    assert adapter.reserve_estimate(task("noul")) == 0.0

    def too_long(_decision):
        raise PromptTooLongError("prompt has 5000 tokens")

    refused = DecPortJevBenchAdapter(too_long, name="t", model="m", upstream=UPSTREAM).run(
        task("choice")
    )
    assert not refused.ok and refused.status == CONTEXT_LIMIT_STATUS
    failed = DecPortJevBenchAdapter(
        lambda _decision: [1.0], name="t", model="m", upstream=UPSTREAM
    ).run(task("choice"))
    assert not failed.ok and failed.status is None and "ValueError" in failed.error


@pytest.mark.skipif(
    not os.environ.get("DECPORT_JEVBENCH_ROOT"),
    reason="set DECPORT_JEVBENCH_ROOT to a clean checkout of the pinned JevBench commit",
)
def test_pinned_public_subset_loads_from_a_clean_checkout() -> None:
    from decport.jevbench import load_jevbench

    upstream = load_jevbench(os.environ["DECPORT_JEVBENCH_ROOT"])

    assert len(upstream.tasks) == 231
    assert {tier for tier in upstream.tier_by_task.values()} == {"original", "easy", "hard"}
    for item in upstream.tasks:
        decision = typed_decision_from_task(item, upstream.base.build_question)
        uniform = [1 / len(decision.options)] * len(decision.options)
        assert set(jevbench_probabilities(decision, uniform, item.labels)) == set(item.labels)
