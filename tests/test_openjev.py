import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from decport.decision_core import LinearDecisionCore, TypedDecision
from decport.openjev import (
    QWEN_BASE_MODEL_ID,
    QWEN_BASE_REVISION,
    OpenJevTeacher,
    load_openjev_core,
    verify_openjev_package,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_package(root: Path, width: int = 6, temperature: float = 1.25):
    """A synthetic package with Open-Jev's layout; the weights are fixture data only."""

    torch.manual_seed(0)
    head = nn.Linear(width, 1)
    checkpoint = root / "checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    torch.save(head.state_dict(), checkpoint / "head.pt")
    (checkpoint / "model.json").write_text(
        json.dumps(
            {
                "model_id": QWEN_BASE_MODEL_ID,
                "revision": QWEN_BASE_REVISION,
                "max_length": 4096,
                "lora_rank": 8,
                "method": "independent_candidate_lora_nll_brier",
            }
        )
    )
    (checkpoint / "temperature.json").write_text(
        json.dumps({"temperature": temperature, "split": "calibration", "n": 4})
    )
    (checkpoint / "adapter" / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"fixture")
    files = {
        path.relative_to(root).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "local_inference_weight_package",
                "model": QWEN_BASE_MODEL_ID,
                "revision": QWEN_BASE_REVISION,
                "files": files,
            }
        )
    )
    return head, _sha256(root / "manifest.json")


def test_frozen_core_loads_the_packaged_head_and_calibration(tmp_path) -> None:
    head, manifest_sha256 = build_package(tmp_path)

    package = verify_openjev_package(tmp_path, expected_manifest_sha256=manifest_sha256)
    core = load_openjev_core(package)

    assert package.temperature == 1.25 and package.lora_rank == 8
    assert core.input_size == 6 and core.temperature == 1.25
    assert torch.equal(core.readout.weight, head.weight.detach())
    assert torch.equal(core.readout.bias, head.bias.detach())
    core.assert_frozen()


def test_package_verification_rejects_tampering_and_other_manifests(tmp_path) -> None:
    _, manifest_sha256 = build_package(tmp_path)

    with pytest.raises(ValueError, match="manifest differs"):
        verify_openjev_package(tmp_path, expected_manifest_sha256="0" * 64)
    torch.save(
        {"weight": torch.zeros(1, 6), "bias": torch.zeros(1)}, tmp_path / "checkpoint" / "head.pt"
    )
    with pytest.raises(ValueError, match="head.pt"):
        verify_openjev_package(tmp_path, expected_manifest_sha256=manifest_sha256)


class FakeOpenJevModel(nn.Module):
    """Mirrors the upstream DecisionModel forward: one head call on last hidden states."""

    def __init__(self, width: int = 6) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.embedding = nn.Embedding(97, width)
        self.head = nn.Linear(width, 1)
        self.model_id, self.revision = QWEN_BASE_MODEL_ID, QWEN_BASE_REVISION
        self.last_input_tokens = 0

    def forward(self, records):
        from jev.api import candidate_prompts

        prompts, counts = [], []
        for record in records:
            rendered = candidate_prompts(record)
            counts.append(len(rendered))
            prompts.extend(rendered)
        identifiers = torch.tensor([sum(map(ord, prompt)) % 97 for prompt in prompts])
        self.last_input_tokens = sum(len(prompt) for prompt in prompts)
        scores = self.head(self.embedding(identifiers).float()).squeeze(-1)
        logits, offset = [], 0
        for record, count in zip(records, counts):
            values = scores[offset : offset + count]
            if record["kind"] == "noul":
                values = torch.stack([torch.zeros_like(values[0]), values[0]])
            logits.append(values)
            offset += count
        return logits


def typed_decisions() -> list[TypedDecision]:
    return [
        TypedDecision(
            "Order arrived broken.", "choice", "Route?", {"billing": "Refunds", "tech": None}
        ),
        TypedDecision("Charged twice.", "noul", "Duplicate charge?"),
        TypedDecision("Service down.", "score", "Urgency?", ["Routine", "Urgent", "Critical"]),
        TypedDecision({"ticket": 7}, "choice", "Pick", {name: None for name in "abcde"}),
    ]


def teacher_and_core(temperature: float = 1.5) -> tuple[OpenJevTeacher, FakeOpenJevModel]:
    model = FakeOpenJevModel()
    core = LinearDecisionCore.from_state_dict(model.head.state_dict(), temperature)
    return OpenJevTeacher(model, core), model


def test_openjev_prompts_keep_upstream_typed_semantics() -> None:
    pytest.importorskip("jev.api")
    from decport.openjev import render_openjev_prompts

    choice, noul, score, _ = typed_decisions()

    assert len(render_openjev_prompts(choice)) == 2
    assert "Proposed answer: billing: Refunds" in render_openjev_prompts(choice)[0]
    (noul_prompt,) = render_openjev_prompts(noul)
    assert noul_prompt.endswith("Is the answer to this question yes? Answer Yes or No.")
    levels = [
        prompt.split("Proposed answer: ")[1].split("\n")[0]
        for prompt in render_openjev_prompts(score)
    ]
    assert levels == ["Routine", "Urgent", "Critical"]
    labeled = TypedDecision("Charged twice.", "noul", "Duplicate charge?", answer="true")
    assert render_openjev_prompts(labeled) == render_openjev_prompts(noul)


def test_teacher_extracts_calibrated_typed_probabilities_with_core_parity() -> None:
    pytest.importorskip("jev.serving")
    from jev.metrics import softmax

    teacher, model = teacher_and_core(temperature=1.5)
    items = typed_decisions()

    batched = teacher.raw_logits(items, candidate_batch_size=8)
    single = teacher.raw_logits(items, candidate_batch_size=1)
    noul_probabilities = teacher.calibrated_probabilities(items[1])

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert [len(row) for row in batched] == [2, 2, 3, 5]
    assert all(torch.allclose(a, b) for a, b in zip(batched, single, strict=True))
    assert batched[1][0].item() == 0.0
    assert teacher.parity_max_abs_diff == pytest.approx(0.0, abs=1e-6)
    assert noul_probabilities == pytest.approx(softmax(batched[1].tolist(), temperature=1.5))
    assert noul_probabilities[1] == pytest.approx(1 / (1 + math.exp(-batched[1][1].item() / 1.5)))
    assert 0.0 < noul_probabilities[1] < 1.0


def test_teacher_rejects_a_different_core_and_labeled_inputs() -> None:
    pytest.importorskip("jev.serving")
    model = FakeOpenJevModel()
    other = LinearDecisionCore(6, 1.5, weight=torch.ones(1, 6), bias=torch.zeros(1))
    with pytest.raises(ValueError, match="differs from the loaded Open-Jev head"):
        OpenJevTeacher(model, other)

    teacher, _ = teacher_and_core()
    labeled = TypedDecision("Charged twice.", "noul", "Duplicate charge?", answer="true")
    with pytest.raises(ValueError, match="must not include answers"):
        teacher.raw_logits([labeled], candidate_batch_size=1)
