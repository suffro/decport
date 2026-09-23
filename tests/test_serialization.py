import json

import pytest
import torch

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.backbones import QwenBackbone, SmolLMBackbone
from decport.serialization import load_artifact, save_artifact
from tests.fakes import FakeCausalLM, FakeTokenizer


def make_model(backbone_type=QwenBackbone, hidden_size: int = 5) -> DecPort:
    backbone = backbone_type(model=FakeCausalLM(hidden_size), tokenizer=FakeTokenizer())
    return DecPort(backbone, BackboneAdapter(hidden_size, 7), DecisionHead(7))


def test_artifact_round_trip(tmp_path) -> None:
    model = make_model()
    path = save_artifact(model, tmp_path / "artifact", include_head=True)
    fresh_backbone = QwenBackbone(model=FakeCausalLM(5), tokenizer=FakeTokenizer())

    loaded = load_artifact(path, fresh_backbone)

    assert_state_dict_equal(model.adapter.state_dict(), loaded.adapter.state_dict())
    assert_state_dict_equal(model.head.state_dict(), loaded.head.state_dict())
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["format_version"] == 1


def test_adapter_only_artifact_reuses_provided_shared_head(tmp_path) -> None:
    model = make_model(SmolLMBackbone)
    path = save_artifact(model, tmp_path / "adapter", include_head=False)
    shared_head = DecisionHead(7)
    fresh_backbone = SmolLMBackbone(model=FakeCausalLM(5), tokenizer=FakeTokenizer())

    loaded = load_artifact(path, fresh_backbone, head=shared_head)

    assert loaded.head is shared_head
    with pytest.raises(ValueError, match="does not include"):
        load_artifact(path, fresh_backbone)


def test_adapter_for_a_frozen_core_reloads_with_identical_outputs(tmp_path) -> None:
    from decport.broad import score_core_batch
    from decport.decision_core import LinearDecisionCore, TypedDecision

    torch.manual_seed(0)
    core = LinearDecisionCore(8, 1.5, weight=torch.randn(1, 8), bias=torch.zeros(1))
    backbone = SmolLMBackbone(model=FakeCausalLM(5), tokenizer=FakeTokenizer())
    model = DecPort(backbone, BackboneAdapter(5, 8, hidden_size=3), core)
    decision = TypedDecision("state", "choice", "Pick", {"a": None, "bb": None})

    def render(item):
        return [f"{item.state}|{option}" for option in item.options]

    path = save_artifact(model, tmp_path / "adapter", include_head=False)
    loaded = load_artifact(path, backbone, head=core)

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["adapter_hidden_size"] == 3 and not config["includes_head"]
    assert loaded.head is core and loaded.adapter.hidden_size == 3
    assert_state_dict_equal(model.adapter.state_dict(), loaded.adapter.state_dict())
    assert torch.equal(
        score_core_batch(model, [decision], render), score_core_batch(loaded, [decision], render)
    )


def test_legacy_artifacts_without_intermediate_width_still_load(tmp_path) -> None:
    model = make_model()
    path = save_artifact(model, tmp_path / "legacy", include_head=True)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    del config["adapter_hidden_size"]
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded = load_artifact(path, QwenBackbone(model=FakeCausalLM(5), tokenizer=FakeTokenizer()))

    assert loaded.adapter.hidden_size == loaded.adapter.shared_size == 7
    assert_state_dict_equal(model.adapter.state_dict(), loaded.adapter.state_dict())


def test_artifact_rejects_wrong_backbone_width(tmp_path) -> None:
    path = save_artifact(make_model(), tmp_path / "artifact", include_head=True)
    wrong_backbone = QwenBackbone(model=FakeCausalLM(6), tokenizer=FakeTokenizer())

    with pytest.raises(ValueError, match="expects hidden size 5"):
        load_artifact(path, wrong_backbone)


def assert_state_dict_equal(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> None:
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
