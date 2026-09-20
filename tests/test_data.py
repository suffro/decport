import json
import random

import pytest

from decport.data import (
    convert_ag_news,
    convert_boolq,
    convert_sst2,
    load_jsonl,
    shuffle_options,
    write_jsonl,
)
from decport.schema import DecisionExample


def test_jsonl_round_trip(tmp_path) -> None:
    examples = [
        DecisionExample("text", "Sentiment?", ("negative", "positive"), "positive"),
        DecisionExample("passage", "Is it true?", ("yes", "no"), "no"),
    ]
    path = tmp_path / "examples.jsonl"

    write_jsonl(examples, path)

    assert load_jsonl(path) == examples


def test_jsonl_error_includes_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"state": "x"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_jsonl(path)


def test_option_shuffle_is_seeded_and_keeps_answer_identity() -> None:
    example = DecisionExample("s", "q", ("a", "b", "c", "d"), "c")

    shuffled = shuffle_options(example, random.Random(7))

    assert shuffled.options == ("d", "b", "a", "c")
    assert shuffled.answer == "c"
    assert shuffled.answer_index == 3


def test_initial_dataset_converters() -> None:
    assert convert_sst2({"sentence": "A wonderful film.", "label": 1}).answer == "positive"
    assert convert_ag_news({"text": "Markets rose.", "label": 2}).answer == "business"
    boolq = {"passage": "Water is wet.", "question": "Is it wet?", "answer": True}
    assert convert_boolq(boolq).answer == "yes"


@pytest.mark.parametrize("label", [-1, 2, "1", True])
def test_sst2_rejects_invalid_labels(label: object) -> None:
    error_type = TypeError if isinstance(label, (str, bool)) else ValueError
    with pytest.raises(error_type):
        convert_sst2({"sentence": "Text", "label": label})
