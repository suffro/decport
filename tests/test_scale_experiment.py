import json

from decport.backbones import GemmaBackbone, QwenBackbone, SmolLMBackbone
from decport.data import write_jsonl
from decport.distillation import DistillationConfig
from decport.scale_experiment import (
    ScaleExperimentConfig,
    TargetSpec,
    aggregate_scale_results,
    run_scale_experiment,
)
from decport.schema import DecisionExample
from decport.train import TrainingConfig
from tests.fakes import FakeCausalLM, FakeTokenizer


def test_scale_experiment_shares_one_source_head_across_targets(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(
        GemmaBackbone,
        "from_pretrained",
        lambda *_args, **_kwargs: GemmaBackbone(FakeCausalLM(6), FakeTokenizer()),
    )
    train = [
        DecisionExample("first", "Pick", ("a", "bb"), "a"),
        DecisionExample("second", "Pick", ("a", "bb"), "bb"),
    ]
    evaluation = [
        DecisionExample("third", "Pick", ("a", "bb"), "a"),
        DecisionExample("fourth", "Pick", ("a", "bb"), "bb"),
    ]
    ood = [
        DecisionExample("fifth", "Pick", ("yes", "no"), "yes"),
        DecisionExample("sixth", "Pick", ("yes", "no"), "no"),
    ]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    ood_path = tmp_path / "ood.jsonl"
    output_path = tmp_path / "output"
    write_jsonl(train, train_path)
    write_jsonl(evaluation, eval_path)
    write_jsonl(ood, ood_path)

    summary = run_scale_experiment(
        ScaleExperimentConfig(
            train_path=str(train_path),
            eval_path=str(eval_path),
            ood_path=str(ood_path),
            output_dir=str(output_path),
            targets=(
                TargetSpec("smollm", "smollm", "fake-smollm"),
                TargetSpec("gemma", "gemma", "fake-gemma"),
            ),
            seeds=(0,),
            shared_size=6,
            cache_batch_size=2,
            device="cpu",
            permutation_trials=0,
            decision_training=TrainingConfig(epochs=1, learning_rate=1e-2),
            distillation_training=DistillationConfig(epochs=1, learning_rate=1e-2),
        )
    )

    result = json.loads((output_path / "seed-0" / "results.json").read_text())
    head_hash = result["source"]["head_sha256"]
    assert summary["seed_count"] == 1
    assert summary["source_head_sha256_by_seed"]["0"] == head_hash
    assert result["label_free_audit"]["distillation_examples_with_answers"] == 0
    assert result["label_free_audit"]["ground_truth_used_for_target_distillation"] is False
    for target in ("smollm", "gemma"):
        target_result = result["targets"][target]
        assert target_result["source_head_sha256"] == head_hash
        assert set(target_result["conditions"]) == {
            "unaligned_target",
            "decision_space_distillation",
            "permuted_teacher_distillation",
            "native_target",
            "supervised_transfer_target",
            "random_head_control",
        }
        assert not (
            output_path
            / "seed-0"
            / "targets"
            / target
            / "decision_space_distillation"
            / "head.safetensors"
        ).exists()
        assert (
            output_path / "seed-0" / "targets" / target / "random_head_control" / "head.safetensors"
        ).is_file()

    doubled = aggregate_scale_results([result, result])
    assert doubled["seed_count"] == 2
    assert (
        doubled["targets"]["gemma"]["metrics"]["decision_space_distillation"]["in_distribution"][
            "accuracy"
        ]["standard_deviation"]
        == 0.0
    )
