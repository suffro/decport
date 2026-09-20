import json

from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.data import write_jsonl
from decport.experiment import ExperimentConfig, run_experiment
from decport.schema import DecisionExample
from decport.train import TrainingConfig
from tests.fakes import FakeCausalLM, FakeTokenizer


def test_three_experiment_runner_writes_observed_results(tmp_path, monkeypatch) -> None:
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
        "decision_portability_ratio",
    }
    assert (output_path / "source" / "head.safetensors").is_file()
    assert (output_path / "native_target" / "head.safetensors").is_file()
    assert not (output_path / "transfer_target" / "head.safetensors").exists()
    assert result["trainable_parameters"]["transfer_target"] > 0  # type: ignore[index]
