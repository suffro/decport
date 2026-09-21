from copy import deepcopy

import torch
from torch import nn

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.backbones import QwenBackbone
from decport.eval import evaluate
from decport.schema import DecisionExample
from decport.train import TrainingConfig, train_decision_model
from tests.fakes import FakeCausalLM, FakeTokenizer


def make_model() -> DecPort:
    backbone = QwenBackbone(model=FakeCausalLM(5), tokenizer=FakeTokenizer())
    return DecPort(backbone, BackboneAdapter(5, 7), DecisionHead(7))


def training_examples() -> list[DecisionExample]:
    return [
        DecisionExample("first", "Pick", ("a", "bb"), "a"),
        DecisionExample("second", "Pick", ("a", "bb"), "bb"),
    ]


def test_source_training_updates_adapter_and_head_but_not_backbone() -> None:
    model = make_model()
    backbone_before = deepcopy(model.backbone.state_dict())
    adapter_before = deepcopy(model.adapter.state_dict())
    head_before = deepcopy(model.head.state_dict())

    history = train_decision_model(
        model,
        training_examples(),
        TrainingConfig(epochs=1, learning_rate=1e-2),
        train_head=True,
    )

    assert len(history.epoch_losses) == 1
    assert history.epoch_losses[0] > 0
    assert_state_dict_equal(backbone_before, model.backbone.state_dict())
    assert_state_dict_changed(adapter_before, model.adapter.state_dict())
    assert_state_dict_changed(head_before, model.head.state_dict())


def test_transfer_training_keeps_shared_head_frozen() -> None:
    model = make_model()
    adapter_before = deepcopy(model.adapter.state_dict())
    head_before = deepcopy(model.head.state_dict())

    train_decision_model(
        model,
        training_examples(),
        TrainingConfig(epochs=1, learning_rate=1e-2),
        train_head=False,
    )

    assert_state_dict_changed(adapter_before, model.adapter.state_dict())
    assert_state_dict_equal(head_before, model.head.state_dict())
    assert all(not parameter.requires_grad for parameter in model.head.parameters())


def test_training_calls_epoch_callback_after_each_epoch() -> None:
    observations: list[tuple[int, float, bool]] = []

    train_decision_model(
        make_model(),
        training_examples(),
        TrainingConfig(epochs=2, learning_rate=1e-2),
        train_head=True,
        epoch_callback=lambda model, epoch, loss: observations.append(
            (epoch, loss, model.training)
        ),
    )

    assert [epoch for epoch, _, _ in observations] == [1, 2]
    assert all(loss > 0 for _, loss, _ in observations)
    assert all(training for _, _, training in observations)


class AnswerFromState(nn.Module):
    training: bool

    def score_options(self, state: str, question: str, options: tuple[str, ...]):
        del question
        return torch.tensor([1.0 if option == state else 0.0 for option in options])

    forward = score_options


class AlwaysFirst(nn.Module):
    training: bool

    def score_options(self, state: str, question: str, options: tuple[str, ...]):
        del state, question
        return torch.tensor([1.0] + [0.0] * (len(options) - 1))

    forward = score_options


def test_evaluation_reports_metrics_and_permutation_robustness() -> None:
    examples = [
        DecisionExample("yes", "q", ("yes", "no"), "yes"),
        DecisionExample("no", "q", ("yes", "no"), "no"),
    ]

    result = evaluate(AnswerFromState(), examples, permutation_trials=4)  # type: ignore[arg-type]

    assert result.example_count == 2
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.option_permutation_robustness == 1.0
    assert result.nll > 0
    assert result.brier > 0


def test_permutation_robustness_never_counts_the_original_order() -> None:
    example = DecisionExample("yes", "q", ("yes", "no"), "yes")

    result = evaluate(AlwaysFirst(), [example], permutation_trials=3)  # type: ignore[arg-type]

    assert result.option_permutation_robustness == 0.0


def assert_state_dict_equal(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> None:
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def assert_state_dict_changed(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> None:
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[key], after[key]) for key in before)
