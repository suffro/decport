import math

import pytest
import torch

from decport import BackboneAdapter, DecPort
from decport.backbones import SmolLMBackbone
from decport.decision_core import LinearDecisionCore, TypedDecision, typed_from_example
from decport.schema import DecisionExample
from tests.fakes import FakeCausalLM, FakeTokenizer

LEVELS = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")


def make_core(width: int = 6, temperature: float = 1.5) -> LinearDecisionCore:
    generator = torch.Generator().manual_seed(0)
    return LinearDecisionCore(
        width,
        temperature,
        weight=torch.randn((1, width), generator=generator),
        bias=torch.tensor([0.25]),
    )


def test_canonical_records_map_to_explicit_choice_noul_and_score() -> None:
    choice = typed_from_example(
        DecisionExample("s", "Which?", ("a", "b", "c"), "b", "arc", "sci", "choice")
    )
    noul = typed_from_example(
        DecisionExample("p", "Is it?", ("yes", "no"), "yes", "boolq", "rc", "boolean")
    )
    score = typed_from_example(
        DecisionExample("r", "Rating?", LEVELS, "4 stars", "yelp", "rating", "score")
    )

    assert (choice.kind, choice.options, choice.answer, choice.candidate_count) == (
        "choice",
        ("a", "b", "c"),
        "b",
        3,
    )
    assert (noul.kind, noul.options, noul.answer, noul.candidate_count) == (
        "noul",
        ("false", "true"),
        "true",
        1,
    )
    assert (score.kind, score.options, score.answer, score.answer_index) == (
        "score",
        ("0", "1", "2", "3", "4"),
        "3",
        3,
    )
    assert score.question() == {
        "type": "score",
        "instructions": "Rating?",
        "criteria": list(LEVELS),
    }
    assert "answer" not in str(noul.question()) and noul.without_answer().answer is None


def test_typed_decision_rejects_invalid_semantics() -> None:
    with pytest.raises(ValueError, match="answer keys"):
        TypedDecision("s", "noul", "Is it?", answer="yes")
    with pytest.raises(ValueError, match="2 to 10"):
        TypedDecision("s", "score", "Rate", criteria=["only"])
    with pytest.raises(ValueError, match="true and false"):
        TypedDecision("s", "noul", "Is it?", criteria={"yes": "y", "no": "n"})
    with pytest.raises(ValueError, match="yes and no"):
        typed_from_example(DecisionExample("s", "q", ("a", "b"), decision_type="boolean"))


def test_core_assembles_typed_logits_with_open_jev_calibration() -> None:
    core = make_core(temperature=1.5)
    scores = torch.tensor([[2.0], [-1.0]])

    noul = core.calibrated_logits("noul", scores)
    choice = core.calibrated_logits("choice", torch.tensor([[3.0, 0.0, -3.0]]))

    assert torch.equal(core.typed_logits("noul", scores), torch.tensor([[0.0, 2.0], [0.0, -1.0]]))
    probabilities = torch.softmax(noul, dim=1)
    assert probabilities[0, 1].item() == pytest.approx(1 / (1 + math.exp(-2.0 / 1.5)))
    assert torch.allclose(choice, torch.tensor([[2.0, 0.0, -2.0]]))
    with pytest.raises(ValueError, match="exactly one candidate"):
        core.typed_logits("noul", torch.zeros((1, 2)))


def test_core_is_frozen_stays_in_eval_and_detects_changes() -> None:
    core = make_core()
    digest = core.state_sha256()

    core.train()

    assert not core.training
    assert all(not parameter.requires_grad for parameter in core.parameters())
    core.assert_frozen()
    core.readout.requires_grad_(True)
    with pytest.raises(RuntimeError, match="must remain frozen"):
        core.assert_frozen()
    with torch.no_grad():
        core.readout.bias.add_(1.0)
    assert core.state_sha256() != digest


def test_random_control_core_keeps_width_norm_bias_and_calibration() -> None:
    core = make_core(width=8)

    first = LinearDecisionCore.random_control(core, seed=3)
    second = LinearDecisionCore.random_control(core, seed=3)

    assert torch.equal(first.readout.weight, second.readout.weight)
    assert not torch.allclose(first.readout.weight, core.readout.weight)
    assert first.readout.weight.norm().item() == pytest.approx(core.readout.weight.norm().item())
    assert torch.equal(first.readout.bias, core.readout.bias)
    assert first.temperature == core.temperature
    first.assert_frozen()


def test_adapter_maps_native_width_into_the_core_width() -> None:
    core = make_core(width=6)
    backbone = SmolLMBackbone(FakeCausalLM(4), FakeTokenizer())
    adapter = BackboneAdapter(4, core.input_size, hidden_size=3)

    model = DecPort(backbone, adapter, core)
    latent = adapter(torch.randn(5, 4))

    assert latent.shape == (5, 6)
    assert core(latent).shape == (5,)
    assert adapter.hidden_size == 3 and adapter.network[1].out_features == 3
    assert model.head is core
    with pytest.raises(ValueError, match="shared sizes do not match"):
        DecPort(backbone, BackboneAdapter(4, 5, hidden_size=3), core)


def test_core_loads_from_an_nn_linear_state_dict() -> None:
    head = torch.nn.Linear(6, 1)

    core = LinearDecisionCore.from_state_dict(head.state_dict(), 2.0)

    assert torch.equal(core.readout.weight, head.weight.detach())
    expected = head(torch.ones(2, 6)).squeeze(-1).tolist()
    assert core(torch.ones(2, 6)).tolist() == pytest.approx(expected)
    with pytest.raises(ValueError, match="exactly weight and bias"):
        LinearDecisionCore.from_state_dict({"weight": head.weight}, 2.0)
