import pytest

from decport.schema import DecisionExample


def test_decision_example_round_trip_and_label_index() -> None:
    raw = {
        "state": "Customer was charged twice.",
        "question": "Which department should handle this?",
        "options": ["billing", "sales", "technical"],
        "answer": "billing",
    }

    example = DecisionExample.from_dict(raw)

    assert example.options == ("billing", "sales", "technical")
    assert example.answer_index == 0
    assert example.to_dict() == raw


@pytest.mark.parametrize(
    ("options", "answer", "message"),
    [
        (("only",), None, "at least two"),
        (("yes", "yes"), None, "unique"),
        (("yes", "no"), "maybe", "exactly match"),
    ],
)
def test_decision_example_rejects_invalid_choices(
    options: tuple[str, ...], answer: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DecisionExample(state="", question="Continue?", options=options, answer=answer)


def test_inference_example_has_no_label_index() -> None:
    example = DecisionExample(state="", question="Continue?", options=("yes", "no"))

    with pytest.raises(ValueError, match="no answer"):
        _ = example.answer_index
