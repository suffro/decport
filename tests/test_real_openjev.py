"""Opt-in check of the real pinned Open-Jev 2B teacher and frozen decision core."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DECPORT_RUN_REAL_MODELS") != "1",
    reason="set DECPORT_RUN_REAL_MODELS=1 to download and run the real Open-Jev 2B teacher",
)


def test_real_openjev_teacher_matches_its_frozen_core() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the Open-Jev 2B teacher test needs CUDA")
    from decport.decision_core import TypedDecision
    from decport.openjev import (
        OPENJEV_COMMIT,
        OpenJevTeacher,
        download_openjev_package,
        installed_openjev_commit,
        load_openjev_core,
        verify_openjev_package,
    )

    assert installed_openjev_commit() == OPENJEV_COMMIT
    package = verify_openjev_package(download_openjev_package())
    core = load_openjev_core(package).to("cuda:0")
    teacher = OpenJevTeacher.from_package(package, core, device="cuda:0")
    decisions = [
        TypedDecision(
            "A customer was charged twice for one order.",
            "choice",
            "Which team should handle this?",
            {"billing": "Charges and refunds", "security": "Account compromise"},
        ),
        TypedDecision("A customer was charged twice for one order.", "noul", "Is a refund due?"),
        TypedDecision(
            "The website is down for every user.",
            "score",
            "How urgent is this?",
            ["Routine", "Urgent", "Critical"],
        ),
    ]

    logits = teacher.raw_logits(decisions, candidate_batch_size=8)

    assert core.input_size == 2048 and core.temperature == pytest.approx(1.518796342858676)
    assert teacher.parity_max_abs_diff <= 1e-5
    assert [len(row) for row in logits] == [2, 2, 3]
    assert logits[1][0].item() == 0.0
    for row in logits:
        probabilities = torch.softmax(row / core.temperature, dim=0)
        assert 0.0 < float(probabilities.max()) < 1.0
