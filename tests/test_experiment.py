import json

import torch

import decport.experiment as experiment_module
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.data import write_jsonl
from decport.experiment import ExperimentConfig, run_experiment
from decport.schema import DecisionExample
from decport.train import TrainingConfig
from tests.fakes import FakeCausalLM, FakeTokenizer


def test_four_experiment_runner_writes_observed_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        QwenBackbone,
        "from_pretrained",
        lambda *_args, **_kwargs: QwenBackbone(FakeCausalLM(5), FakeTokenizer()),
    )
    monkeypatch.setattr(
        SmolLMBackbone,
        "from_pretrained",
        lambda *_args, **_kwargs: SmolLMBackbone(FakeCausalLM(4), FakeTokenizer()),
    )
    initial_adapters: list[dict[str, torch.Tensor]] = []
    initial_heads: list[dict[str, torch.Tensor]] = []
    heads_unchanged: list[bool] = []
    heads_trainable_after: list[bool] = []
    train_head_values: list[bool] = []
    real_train = experiment_module.train_decision_model

    def capture_initial_adapter(*args, **kwargs):
        model = args[0]
        head_before = {
            key: value.detach().clone() for key, value in model.head.state_dict().items()
        }
        initial_heads.append(head_before)
        initial_adapters.append(
            {key: value.detach().clone() for key, value in model.adapter.state_dict().items()}
        )
        train_head_values.append(kwargs["train_head"])
        history = real_train(*args, **kwargs)
        heads_unchanged.append(
            all(
                torch.equal(head_before[key], model.head.state_dict()[key])
                for key in head_before
            )
        )
        heads_trainable_after.append(
            any(parameter.requires_grad for parameter in model.head.parameters())
        )
        return history

    monkeypatch.setattr(experiment_module, "train_decision_model", capture_initial_adapter)
    examples = [
        DecisionExample("first", "Pick", ("a", "bb"), "a"),
        DecisionExample("second", "Pick", ("a", "bb"), "bb"),
    ]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "output"
    write_jsonl(examples, train_path)
    write_jsonl(examples, eval_path)

    result = run_experiment(
        ExperimentConfig(
            train_path=str(train_path),
            eval_path=str(eval_path),
            output_dir=str(output_path),
            shared_size=6,
            device="cpu",
            permutation_trials=1,
            training=TrainingConfig(epochs=1, learning_rate=1e-2),
        )
    )

    persisted = json.loads((output_path / "results.json").read_text(encoding="utf-8"))
    assert persisted == result
    assert set(result) >= {
        "source",
        "native_target",
        "transfer_target",
        "random_head_control",
        "decision_portability_ratio",
        "decport_accuracy_gain_over_random_head",
    }
    assert (output_path / "source" / "head.safetensors").is_file()
    assert (output_path / "native_target" / "head.safetensors").is_file()
    assert not (output_path / "transfer_target" / "head.safetensors").exists()
    assert (output_path / "random_head_control" / "head.safetensors").is_file()
    assert result["trainable_parameters"]["transfer_target"] > 0  # type: ignore[index]
    for target_run in (2, 3):
        assert initial_adapters[1].keys() == initial_adapters[target_run].keys()
        assert all(
            torch.equal(initial_adapters[1][key], initial_adapters[target_run][key])
            for key in initial_adapters[1]
        )
    assert train_head_values == [True, True, False, False]
    assert any(
        not torch.equal(initial_heads[2][key], initial_heads[3][key])
        for key in initial_heads[2]
    )
    assert heads_unchanged[3]
    assert not heads_trainable_after[3]
