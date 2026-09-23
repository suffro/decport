import copy

import pytest
import torch

from decport import DecPort
from decport.adapter import LowRankAdapter
from decport.backbones import SmolLMBackbone
from decport.broad import predict_core_logits, train_core_distillation_batched
from decport.decision_core import MLPDecisionCore, ScalarIdentityCore, TypedDecision
from decport.distillation import DistillationConfig, TeacherOutput
from decport.final_gate import (
    CONDITIONS,
    SourceCoreConfig,
    TargetDecisionModule,
    _content_overlap,
    core_calibrated_logits,
    evaluate_ship_gate,
    fit_temperature,
    select_jevbench_seeds,
    train_source_core,
)
from decport.final_gate import (
    _paired as paired,
)
from tests.fakes import FakeCausalLM, FakeTokenizer
from tests.test_core_transfer import decisions, render, teacher_outputs


def mlp_core(width: int = 8, temperature: float = 1.5, seed: int = 0) -> MLPDecisionCore:
    torch.manual_seed(seed)
    from decport.decision_core import mlp_decision_network

    state = mlp_decision_network(width, (6, 4)).state_dict()
    return MLPDecisionCore(width, temperature, state=state, hidden_sizes=(6, 4))


def test_mlp_core_has_the_frozen_protocol_architecture() -> None:
    from decport.decision_core import mlp_decision_network

    state = mlp_decision_network(2048).state_dict()
    core = MLPDecisionCore(2048, 1.3, state=state)

    assert core.parameter_count == 1_118_977
    assert [type(layer).__name__ for layer in core.network] == [
        "LayerNorm", "Linear", "GELU", "Linear", "GELU", "Linear",
    ]
    assert not any(parameter.requires_grad for parameter in core.parameters())
    assert core(torch.randn(3, 2048)).shape == (3,)
    core.train()
    assert not core.training


def test_random_core_matches_layer_scale_without_the_learned_direction() -> None:
    core = mlp_core()
    with torch.no_grad():  # a "trained" core whose scale comes from aligned layers
        core.network[1].weight.add_(0.5 * torch.ones_like(core.network[1].weight))
        core.network[5].weight.mul_(20.0)
    inputs = torch.randn(64, 8, generator=torch.Generator().manual_seed(1))
    random_core, statistics = MLPDecisionCore.random_control(core, seed=11, representations=inputs)
    again, _ = MLPDecisionCore.random_control(core, seed=11, representations=inputs)
    other, _ = MLPDecisionCore.random_control(core, seed=12, representations=inputs)

    learned_input = random_input = inputs
    linear = 0
    for learned_layer, random_layer in zip(core.network, random_core.network, strict=True):
        learned_input, random_input = learned_layer(learned_input), random_layer(random_input)
        if isinstance(random_layer, torch.nn.Linear):
            assert float(random_input.std()) == pytest.approx(float(learned_input.std()), rel=1e-4)
            assert float(random_input.mean()) == pytest.approx(
                float(learned_input.mean()), abs=1e-4 * float(learned_input.std())
            )
            assert not torch.allclose(random_layer.weight, learned_layer.weight)
            linear += 1
    assert linear == len(statistics["linear_layers"]) == 3
    assert random_core.temperature == core.temperature
    assert random_core.state_sha256() == again.state_sha256() != other.state_sha256()
    assert random_core.state_sha256() != core.state_sha256()
    assert not any(parameter.requires_grad for parameter in random_core.parameters())


@pytest.mark.parametrize(
    ("width", "adapter", "module"),
    [(960, 386_944, 559_745), (640, 345_344, 395_265), (2048, 528_384, 1_118_977)],
)
def test_protocol_parameter_counts(width: int, adapter: int, module: int) -> None:
    low_rank = LowRankAdapter(width, 2048, rank=128)

    assert low_rank.parameter_count == adapter
    assert TargetDecisionModule(width).parameter_count == module
    linear_layers = [layer for layer in low_rank.network if isinstance(layer, torch.nn.Linear)]
    assert [layer.bias for layer in linear_layers] == [None, None]
    assert [type(layer).__name__ for layer in low_rank.network] == [
        "LayerNorm", "Linear", "Linear",
    ]


def test_low_rank_adapter_is_linear_after_its_input_normalization() -> None:
    adapter = LowRankAdapter(5, 7, rank=3)
    normalized = adapter.network[0](torch.randn(4, 5))
    mapped = adapter.network[2](adapter.network[1](normalized))
    combined = adapter.network[2].weight @ adapter.network[1].weight

    assert torch.allclose(mapped, normalized @ combined.T, atol=1e-6)
    assert torch.linalg.matrix_rank(combined) <= 3


def fake_backbone() -> SmolLMBackbone:
    return SmolLMBackbone(FakeCausalLM(4), FakeTokenizer())


def test_target_specific_module_trains_through_the_typed_path_without_a_shared_core() -> None:
    torch.manual_seed(0)
    module = TargetDecisionModule(4, (6, 3))
    model = DecPort(fake_backbone(), module, ScalarIdentityCore(1.5))
    items = decisions()
    before = copy.deepcopy(module.state_dict())

    history = train_core_distillation_batched(
        model, items, teacher_outputs(items), DistillationConfig(epochs=2, temperature=1.0),
        render=render, batch_size=2,
    )
    logits = predict_core_logits(model, items, render=render, batch_size=4)

    assert history.gradient_audit["core_tensors_with_gradient"] == 0
    assert history.gradient_audit["adapter_tensors_with_gradient"] == len(before)
    assert any(not torch.equal(value, module.state_dict()[key]) for key, value in before.items())
    assert [len(row) for row in logits] == [len(item.options) for item in items]
    assert logits[1][0] == 0.0  # Noul keeps [0, s]


def labeled_source(count: int = 12) -> tuple[list[TypedDecision], list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(3)
    items, reps = [], []
    for index in range(count):
        kind = ("choice", "noul", "score")[index % 3]
        if kind == "choice":
            item = TypedDecision(f"s{index}", "choice", "Pick", {"a": None, "b": None},
                                 answer="a", dataset="arc")
        elif kind == "noul":
            item = TypedDecision(f"s{index}", "noul", "True?",
                                 answer=("true", "false")[index % 2], dataset="boolq")
        else:
            item = TypedDecision(f"s{index}", "score", "Rate", ("low", "mid", "high"),
                                 answer="2", dataset="yelp")
        block = torch.randn((item.candidate_count, 8), generator=generator)
        if kind == "choice":
            block[0, 0] += 4.0
        if kind == "noul":
            block[0, 0] += 4.0 if item.answer == "true" else -4.0
        if kind == "score":
            block[2, 0] += 4.0
        items.append(item)
        reps.append(block)
    return items, reps


def test_source_core_training_uses_labels_and_rejects_unlabeled_records() -> None:
    items, reps = labeled_source()
    config = SourceCoreConfig(epochs=30, batch_size=4, learning_rate=1e-2)

    first = train_source_core(
        items, reps, config, input_size=8, hidden_sizes=(6, 4), device=torch.device("cpu")
    )
    second = train_source_core(
        items, reps, config, input_size=8, hidden_sizes=(6, 4), device=torch.device("cpu")
    )

    assert first.epoch_losses[-1] < first.epoch_losses[0]
    assert first.epoch_accuracy[-1] == 1.0
    assert all(torch.equal(value, second.state[key]) for key, value in first.state.items())
    with pytest.raises(ValueError, match="labeled source decisions"):
        train_source_core(
            [item.without_answer() for item in items], reps, config, input_size=8,
            hidden_sizes=(6, 4), device=torch.device("cpu"),
        )


def test_core_logits_are_calibrated_and_typed() -> None:
    items, reps = labeled_source()
    core = mlp_core(temperature=2.0)
    unit = mlp_core(temperature=1.0)

    calibrated = core_calibrated_logits(core, items, reps, batch_size=2)
    raw = core_calibrated_logits(unit, items, reps)

    assert all(torch.allclose(a * 2.0, b, atol=1e-6) for a, b in zip(calibrated, raw))
    assert calibrated[1][0] == 0.0 and len(calibrated[1]) == 2


def test_temperature_fit_recovers_a_known_scale() -> None:
    generator = torch.Generator().manual_seed(0)
    items, logits = [], []
    for index in range(3000):
        base = torch.randn(3, generator=generator)
        label = int(torch.multinomial(torch.softmax(base, dim=0), 1, generator=generator))
        items.append(TypedDecision(f"x{index}", "score", "Rate", ("a", "b", "c"),
                                   answer=str(label), dataset="d"))
        logits.append(base * 3.0)

    fitted = fit_temperature(items, logits, (0.05, 20.0, 4001))

    assert fitted["temperature"] == pytest.approx(3.0, rel=0.1)
    assert fitted["calibration_nll"] <= fitted["calibration_nll_at_temperature_1"]


def test_source_and_transfer_overlap_is_detected() -> None:
    items = decisions()
    assert _content_overlap(items[:2], items[2:]) == 0
    assert _content_overlap(items[:2], items[1:3]) == 1


def seed_results(effects: dict[str, dict[str, float]], seeds=(0, 1, 2, 3, 4), noise=0.0):
    """Synthetic per-seed results: accuracy per (target, condition, split)."""

    results = []
    for seed in seeds:
        targets = {}
        for target, values in effects.items():
            conditions = {}
            for condition in CONDITIONS:
                metrics = {}
                for split in ("in_distribution", "out_of_distribution"):
                    accuracy = values[f"{condition}:{split}"] + noise * (seed - 2)
                    metrics[split] = {"overall": {"accuracy": accuracy}}
                conditions[condition] = {
                    "metrics": metrics,
                    "training": None if condition == "untrained_adapter" else {
                        "epoch_losses": [1.0, 0.5 + 0.01 * ((seed * 7 + 1) % 5)]
                    },
                }
            targets[target] = {
                "conditions": conditions,
                "parameters": {"adapter": values["adapter"],
                               "target_specific_module": values["module"]},
            }
        results.append({"seed": seed, "targets": targets})
    return results


def effect(learned_id, learned_ood, *, random_id=0.30, random_ood=0.30, baseline_id=0.30,
           baseline_ood=0.30, adapter=386_944, module=559_745):
    return {
        "learned_core_distillation:in_distribution": learned_id,
        "learned_core_distillation:out_of_distribution": learned_ood,
        "random_core_distillation:in_distribution": random_id,
        "random_core_distillation:out_of_distribution": random_ood,
        "mismatched_teacher_distillation:in_distribution": 0.25,
        "mismatched_teacher_distillation:out_of_distribution": 0.25,
        "untrained_adapter:in_distribution": 0.25,
        "untrained_adapter:out_of_distribution": 0.25,
        "target_specific_distillation:in_distribution": baseline_id,
        "target_specific_distillation:out_of_distribution": baseline_ood,
        "adapter": adapter,
        "module": module,
    }


JEVBENCH_PASS = {name: {"n_correct": 90, "n_planned": 231, "uniform_expected": 73.37}
                 for name in ("smollm", "gemma", "tinyllama")}


def test_ship_requires_every_criterion_audit_and_reproduction() -> None:
    results = seed_results({name: effect(0.45, 0.40) for name in ("smollm", "gemma", "tinyllama")})

    gate = evaluate_ship_gate(results, JEVBENCH_PASS, audit_passed=True, reproduction_passed=True)
    assert gate["verdict"] == "SHIP"
    assert all(gate["criteria"].values())

    no_repro = evaluate_ship_gate(results, JEVBENCH_PASS, audit_passed=True,
                                  reproduction_passed=False)
    assert no_repro["verdict"] == "NO-SHIP"
    missing = evaluate_ship_gate(results, None, audit_passed=True, reproduction_passed=True)
    assert missing["verdict"] == "NO-SHIP" and not missing["criteria"]["D_external_behavior"]


def test_learned_core_equal_to_random_core_is_no_ship() -> None:
    results = seed_results({
        name: effect(0.45, 0.40, random_id=0.44, random_ood=0.39)
        for name in ("smollm", "gemma", "tinyllama")
    })

    gate = evaluate_ship_gate(results, JEVBENCH_PASS, audit_passed=True, reproduction_passed=True)

    assert gate["verdict"] == "NO-SHIP"
    assert not gate["criteria"]["A_learned_core_matters"]
    assert gate["targets_passing"]["A_learned_core_matters"] == []


def test_one_reusing_backbone_or_a_clearly_negative_one_is_no_ship() -> None:
    one = seed_results({
        "smollm": effect(0.45, 0.40),
        "gemma": effect(0.45, 0.40, random_id=0.44, random_ood=0.39),
        "tinyllama": effect(0.45, 0.40, random_id=0.44, random_ood=0.39),
    })
    negative = seed_results({
        "smollm": effect(0.45, 0.40),
        "gemma": effect(0.45, 0.40),
        "tinyllama": effect(0.45, 0.40, random_id=0.48, random_ood=0.30),
    })

    assert not evaluate_ship_gate(one, JEVBENCH_PASS, audit_passed=True,
                                  reproduction_passed=True)["criteria"]["A_learned_core_matters"]
    gate = evaluate_ship_gate(negative, JEVBENCH_PASS, audit_passed=True, reproduction_passed=True)
    assert gate["targets_passing"]["A_learned_core_matters"] == ["smollm", "gemma"]
    assert not gate["criteria"]["A_learned_core_matters"]


def test_practical_utility_match_route_needs_half_the_parameters() -> None:
    close = {"baseline_id": 0.46, "baseline_ood": 0.41}
    results = seed_results({
        "smollm": effect(0.45, 0.40, **close),
        "gemma": effect(0.45, 0.40, **close, adapter=345_344, module=395_265),
        "tinyllama": effect(0.45, 0.40, **close, adapter=528_384, module=1_118_977),
    })

    gate = evaluate_ship_gate(results, JEVBENCH_PASS, audit_passed=True, reproduction_passed=True)

    assert gate["targets_passing"]["E_practical_utility"] == ["tinyllama"]
    assert not gate["criteria"]["E_practical_utility"]
    assert gate["verdict"] == "NO-SHIP"


def test_positive_seed_count_is_enforced() -> None:
    # Mean gain is large, but three of five seeds are negative.
    results = seed_results({name: effect(0.45, 0.40) for name in ("smollm", "gemma", "tinyllama")})
    for result in results[:3]:
        for target in result["targets"].values():
            target["conditions"]["random_core_distillation"]["metrics"]["in_distribution"][
                "overall"]["accuracy"] = 0.46
    for result in results[3:]:
        for target in result["targets"].values():
            target["conditions"]["random_core_distillation"]["metrics"]["in_distribution"][
                "overall"]["accuracy"] = 0.0

    gate = evaluate_ship_gate(results, JEVBENCH_PASS, audit_passed=True, reproduction_passed=True)

    per_target = gate["per_target"]["smollm"]["A_learned_core_matters"]
    assert per_target["id_gain_over_random_core"]["mean"] > 0.05
    assert per_target["id_gain_over_random_core"]["positive_seeds"] == 2
    assert not per_target["passes"]


def test_jevbench_selection_is_label_free_and_fixed() -> None:
    results = seed_results({name: effect(0.45, 0.40) for name in ("smollm", "gemma", "tinyllama")})
    for result in results:
        result["targets"]["gemma"]["conditions"]["learned_core_distillation"]["training"][
            "epoch_losses"] = [1.0, 0.5]

    selection = select_jevbench_seeds(results)

    losses = {result["seed"]: 0.5 + 0.01 * ((result["seed"] * 7 + 1) % 5) for result in results}
    expected = min(losses, key=lambda seed: (losses[seed], seed))
    assert expected == 2
    assert selection["smollm"]["learned_core_distillation"] == expected
    assert selection["gemma"]["learned_core_distillation"] == 0  # tie -> lowest seed
    assert selection["tinyllama"]["untrained_adapter"] == 0


def test_paired_differences_count_positive_seeds() -> None:
    summary = paired([0.1, -0.2, 0.3, 0.0, 0.05])
    assert summary["positive_seeds"] == 3
    assert summary["mean"] == pytest.approx(0.05)


def test_teacher_outputs_for_scalar_identity_core_keep_calibration() -> None:
    core = ScalarIdentityCore(2.0)
    scores = torch.tensor([[1.0], [-2.0]])
    assert torch.equal(core.calibrated_logits("noul", core(scores).reshape(2, 1)),
                       torch.tensor([[0.0, 0.5], [0.0, -1.0]]))
    assert list(core.parameters()) == []
    assert TeacherOutput(torch.zeros(2)).scores.shape == (2,)
