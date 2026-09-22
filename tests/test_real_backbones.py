"""Opt-in smoke test for the actual v0.1 Hugging Face backbones."""

import os

import pytest

from decport import BackboneAdapter, DecisionHead, DecPort
from decport.backbones import GemmaBackbone, QwenBackbone, SmolLMBackbone

pytestmark = pytest.mark.skipif(
    os.environ.get("DECPORT_RUN_REAL_MODELS") != "1",
    reason="set DECPORT_RUN_REAL_MODELS=1 to download and test real backbones",
)


@pytest.mark.parametrize("backbone_type", [QwenBackbone, SmolLMBackbone, GemmaBackbone])
def test_real_backbone_choice(backbone_type: type) -> None:
    backbone = backbone_type.from_pretrained(model_kwargs={"dtype": "auto"})
    model = DecPort(
        backbone=backbone,
        adapter=BackboneAdapter(backbone.hidden_size, shared_size=16),
        head=DecisionHead(shared_size=16),
    )

    result = model.choice("A duplicate charge appeared.", "Route this", ["billing", "sales"])

    assert result["choice"] in {"billing", "sales"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
