import json

import decport.alignment_experiment as experiment_module
from decport.alignment import AlignmentConfig
from decport.alignment_experiment import (
    AlignmentExperimentConfig,
    aggregate_distillation_results,
    run_alignment_experiment,
)
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.data import write_jsonl
from decport.distillation import DistillationConfig
from decport.schema import DecisionExample
from decport.train import TrainingConfig
from tests.fakes import FakeCausalLM, FakeTokenizer


def test_alignment_experiment_records_controls_and_label_free_audit(
    tmp_path, monkeypatch
) -> None:
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
    observed_alignment_examples = []
    observed_distillation_examples = []
    real_alignment = experiment_module.train_latent_alignment

    def capture_alignment(source, target, examples, config, **kwargs):
        observed_alignment_examples.extend(examples)
        return real_alignment(source, target, examples, config, **kwargs)

    monkeypatch.setattr(
        experiment_module, "train_latent_alignment", capture_alignment
    )
    real_distillation = experiment_module.train_decision_distillation

    def capture_distillation(teacher, student, examples, outputs, config):
        observed_distillation_examples.extend(examples)
        return real_distillation(teacher, student, examples, outputs, config)

    monkeypatch.setattr(
        experiment_module, "train_decision_distillation", capture_distillation
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

    result = run_alignment_experiment(
        AlignmentExperimentConfig(
            train_path=str(train_path),
            eval_path=str(eval_path),
            output_dir=str(output_path),
            shared_size=6,
            device="cpu",
            permutation_trials=0,
            decision_training=TrainingConfig(epochs=1, learning_rate=1e-2),
            alignment_training=AlignmentConfig(epochs=1, learning_rate=1e-2),
            alignment_methods=(
                "cosine_mse",
                "whitened_cosine_mse",
                "ridge",
                "orthogonal_procrustes",
            ),
            distillation_training=DistillationConfig(
                epochs=1, learning_rate=1e-2
            ),
        )
    )

    assert json.loads((output_path / "results.json").read_text()) == result
    assert all(example.answer is None for example in observed_alignment_examples)
    assert all(example.answer is None for example in observed_distillation_examples)
    assert result["label_free_audit"]["alignment_examples_with_answers"] == 0
    assert result["label_free_audit"]["decision_loss_used"] is False
    assert result["label_free_audit"]["distillation_decision_loss_used"] is False
    assert result["label_free_audit"]["distillation_examples_with_answers"] == 0
    assert result["label_free_audit"]["distillation_teacher_outputs_detached"] is True
    assert all(
        f"{method['artifact']}.decision_head"
        in result["label_free_audit"]["frozen_components"]
        for method in result["alignment_methods"].values()
    )
    assert set(result) >= {
        "source",
        "native_target",
        "supervised_transfer_target",
        "random_head_control",
        "unaligned_target",
        "label_free_aligned_target",
        "comparisons",
        "alignment_methods",
        "distillation_conditions",
    }
    assert set(result["alignment_methods"]) == {
        "cosine_mse",
        "whitened_cosine_mse",
        "ridge",
        "orthogonal_procrustes",
    }
    assert len(
        result["label_free_aligned_target"]["training"]["epoch_metrics"]
    ) == 1
    assert set(result["distillation_conditions"]) == {
        "decision_space_distillation",
        "permuted_teacher_distillation",
    }
    assert set(result["comparisons"]["in_distribution"]) >= {
        "distillation_gain_over_unaligned",
        "distillation_gain_over_latent_alignment",
        "distillation_gain_over_permuted_teacher",
    }
    assert set(result["comparisons"]["in_distribution"]["dpr_vs_native_smollm"]) >= {
        "source",
        "native_target",
        "latent_alignment",
        "decision_space_distillation",
        "permuted_teacher_distillation",
    }
    assert result["distillation_objective"]["temperature"] == 2.0
    assert set(result["comparisons"]["in_distribution"]) >= {
        "label_free_alignment_dpr",
        "label_free_gain_over_unaligned",
        "label_free_gain_over_random_head",
    }
    assert (output_path / "source" / "head.safetensors").is_file()
    assert not (
        output_path / "label_free_aligned_target" / "head.safetensors"
    ).exists()
    assert (output_path / "random_head_control" / "head.safetensors").is_file()
    for method in result["alignment_methods"].values():
        assert not (output_path / method["artifact"] / "head.safetensors").exists()
    for condition in result["distillation_conditions"].values():
        assert not (output_path / condition["artifact"] / "head.safetensors").exists()

    summary = aggregate_distillation_results([result, result])
    assert summary["seed_count"] == 2
    assert (
        summary["metrics"]["decision_space_distillation"]["in_distribution"][
            "accuracy"
        ]["standard_deviation"]
        == 0.0
    )
    assert len(summary["training"]["decision_space_distillation"]) == 1
