"""Pinned Open-Jev 2B teacher and its frozen decision core.

Open-Jev is an external system. Its model loader, typed-request compiler, candidate prompt renderer,
and candidate batching come from the pinned ``open-jev`` package (``uv sync --extra openjev``);
DecPort does not reimplement or retrain any of them. This module only verifies the published
package, exposes its trained head plus calibration as a :class:`LinearDecisionCore`, and records
teacher outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from decport.backbones.base import PromptTooLongError
from decport.decision_core import LinearDecisionCore, TypedDecision

OPENJEV_REPOSITORY = "https://github.com/Zefan-Cai/Open-Jev"
OPENJEV_COMMIT = "3308a15ccd7eea1df7a37d6ddc39b023b801ba16"
OPENJEV_2B_REPO_ID = "ZefanCai/Open-Jev-2B"
OPENJEV_2B_REVISION = "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
OPENJEV_2B_MANIFEST_SHA256 = "58319da5c2a948a4645e46d9c982be44867d78779ea1c3bfb81b64867f58ef3a"
QWEN_BASE_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CHECKPOINT_FILES = (
    "checkpoint/adapter/adapter_config.json",
    "checkpoint/adapter/adapter_model.safetensors",
    "checkpoint/head.pt",
    "checkpoint/model.json",
    "checkpoint/temperature.json",
)
# The DecPort core and the upstream head run the same float32 linear op on the same captured input.
PARITY_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class OpenJevPackage:
    """A verified local Open-Jev inference package (not including base weights)."""

    root: Path
    manifest_sha256: str
    model_id: str
    revision: str
    lora_rank: int
    max_length: int
    temperature: float
    file_sha256: dict[str, str] = field(default_factory=dict)

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint"

    def provenance(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "base_model_id": self.model_id,
            "base_revision": self.revision,
            "lora_rank": self.lora_rank,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "file_sha256": dict(sorted(self.file_sha256.items())),
        }


def download_openjev_package(
    repo_id: str = OPENJEV_2B_REPO_ID,
    revision: str = OPENJEV_2B_REVISION,
) -> Path:
    """Download only the inference package from one pinned Hub revision."""

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id, revision=revision, allow_patterns=["package/*"])
    return Path(snapshot) / "package"


def verify_openjev_package(
    root: str | Path,
    *,
    expected_manifest_sha256: str = OPENJEV_2B_MANIFEST_SHA256,
    expected_model_id: str = QWEN_BASE_MODEL_ID,
    expected_revision: str = QWEN_BASE_REVISION,
) -> OpenJevPackage:
    """Verify the package manifest, every listed file, the base pin, and the calibration."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Open-Jev package manifest differs from the pinned SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "local_inference_weight_package"
        or manifest.get("model") != expected_model_id
        or manifest.get("revision") != expected_revision
    ):
        raise ValueError("Open-Jev package kind or base model pin differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(CHECKPOINT_FILES) <= set(files):
        raise ValueError("Open-Jev package lacks required checkpoint files")
    hashes = {}
    for name, record in sorted(files.items()):
        path = root / name
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Open-Jev package file is missing or has the wrong size: {name}")
        hashes[name] = _sha256(path)
        if hashes[name] != record.get("sha256"):
            raise ValueError(f"Open-Jev package file checksum differs: {name}")
    config = json.loads((root / "checkpoint/model.json").read_text(encoding="utf-8"))
    if config.get("model_id") != expected_model_id or config.get("revision") != expected_revision:
        raise ValueError("Open-Jev checkpoint base model pin differs")
    lora_rank, max_length = config.get("lora_rank"), config.get("max_length")
    if type(lora_rank) is not int or lora_rank < 1 or type(max_length) is not int:
        raise ValueError("Open-Jev checkpoint LoRA rank or max length is invalid")
    calibration = json.loads((root / "checkpoint/temperature.json").read_text(encoding="utf-8"))
    temperature = calibration.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
        or calibration.get("split") != "calibration"
    ):
        raise ValueError("Open-Jev calibration temperature is invalid")
    return OpenJevPackage(
        root=root,
        manifest_sha256=manifest_sha256,
        model_id=expected_model_id,
        revision=expected_revision,
        lora_rank=lora_rank,
        max_length=max_length,
        temperature=float(temperature),
        file_sha256=hashes,
    )


def load_openjev_core(package: OpenJevPackage) -> LinearDecisionCore:
    """Load the trained Open-Jev head and saved temperature as one frozen decision core."""

    state = torch.load(package.checkpoint / "head.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Open-Jev head.pt must contain a state dict")
    return LinearDecisionCore.from_state_dict(state, package.temperature)


def compile_openjev_record(decision: TypedDecision) -> dict[str, object]:
    """Compile one typed decision with Open-Jev's own request compiler."""

    from jev.api import compile_request

    (record,) = compile_request(decision.state, {"decision": decision.question()})
    if tuple(record["answer_keys"]) != decision.options:
        raise ValueError("Open-Jev answer keys differ from the typed decision's keys")
    return record


def render_openjev_prompts(decision: TypedDecision) -> list[str]:
    """Open-Jev candidate prompts (before any chat template); one per scored candidate."""

    from jev.api import candidate_prompts

    prompts = candidate_prompts(compile_openjev_record(decision))
    if len(prompts) != decision.candidate_count:
        raise ValueError("Open-Jev rendered an unexpected number of candidate prompts")
    return prompts


class OpenJevTeacher:
    """The complete frozen Open-Jev model, used only to produce teacher distributions.

    Logits come from the upstream forward pass. The exact tensor entering the upstream head is
    captured and re-scored by the shared DecPort core; any mismatch aborts, so the core used by
    every target is proven to be the teacher's decision head.
    """

    def __init__(self, model, core: LinearDecisionCore) -> None:
        model.requires_grad_(False)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("the Open-Jev teacher must be completely frozen")
        core.assert_frozen()
        head = model.head
        if not (
            torch.equal(head.weight.detach().float().cpu(), core.readout.weight.detach().cpu())
            and torch.equal(head.bias.detach().float().cpu(), core.readout.bias.detach().cpu())
        ):
            raise ValueError("the decision core differs from the loaded Open-Jev head")
        self.model = model
        self.core = core
        self.parity_max_abs_diff = 0.0
        self.input_tokens = 0
        self.candidate_sequences = 0

    @classmethod
    def from_package(
        cls,
        package: OpenJevPackage,
        core: LinearDecisionCore,
        *,
        device: str = "cuda:0",
        max_length: int | None = None,
    ) -> OpenJevTeacher:
        from jev.model import DecisionModel

        model = DecisionModel.load(package.checkpoint, device=device)
        if model.model_id != package.model_id or model.revision != package.revision:
            raise ValueError("loaded Open-Jev model identity differs from the package")
        if max_length is not None:
            model.max_length = max_length
        return cls(model, core)

    @property
    def temperature(self) -> float:
        return self.core.temperature

    def raw_logits(
        self,
        decisions: Sequence[TypedDecision],
        *,
        candidate_batch_size: int,
    ) -> list[Tensor]:
        """Upstream typed logits for each decision, in input order.

        Candidate batching uses Open-Jev's own ``candidate_batches``; large Choices are split and
        their logits reassembled before any normalization, exactly as the upstream service does.
        """

        return self.raw_logits_and_representations(
            decisions, candidate_batch_size=candidate_batch_size
        )[0]

    def raw_logits_and_representations(
        self,
        decisions: Sequence[TypedDecision],
        *,
        candidate_batch_size: int,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Upstream typed logits plus the exact float32 head inputs, one row per candidate.

        The representations are the tensors the upstream head reads, captured by a forward hook and
        parity-checked against the upstream logits on every batch; ``(candidates, hidden)`` per
        decision, with one candidate for Noul.
        """

        from jev.serving import candidate_batches

        if any(decision.answer is not None for decision in decisions):
            raise ValueError("teacher inputs must not include answers")
        records = [compile_openjev_record(decision) for decision in decisions]
        order = sorted(range(len(records)), key=lambda index: _record_length(records[index]))
        rows: list[list[float]] = [[] for _ in records]
        states: list[list[Tensor]] = [[] for _ in records]
        ordered = [records[index] for index in order]
        for batch in candidate_batches(ordered, candidate_batch_size):
            pieces = [piece for _, piece in batch]
            scored, captured = self._score_pieces(pieces)
            for (position, _), row, hidden in zip(batch, scored, captured, strict=True):
                rows[order[position]].extend(row)
                states[order[position]].append(hidden)
        logits = [torch.tensor(row, dtype=torch.float32) for row in rows]
        representations = [torch.cat(parts) for parts in states]
        for decision, row, hidden in zip(decisions, logits, representations, strict=True):
            if len(row) != len(decision.options) or not torch.isfinite(row).all():
                raise RuntimeError("Open-Jev returned invalid typed logits")
            if hidden.shape[0] != decision.candidate_count or not torch.isfinite(hidden).all():
                raise RuntimeError("Open-Jev returned invalid candidate representations")
        return logits, representations

    def adapter_and_head_sha256(self) -> str:
        """Digest of the trained LoRA tensors and the head: the non-base weights of the teacher."""

        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            if "lora_" in name or name.startswith("head."):
                digest.update(name.encode("utf-8"))
                value = parameter.detach().cpu().contiguous().reshape(-1)
                digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def calibrated_probabilities(self, decision: TypedDecision) -> list[float]:
        """Open-Jev's calibrated distribution for one decision, as its service returns it."""

        from jev.metrics import softmax

        (row,) = self.raw_logits([decision], candidate_batch_size=1)
        return softmax(row.tolist(), temperature=self.temperature)

    def _score_pieces(
        self, pieces: list[dict[str, object]]
    ) -> tuple[list[list[float]], list[Tensor]]:
        captured: list[Tensor] = []
        handle = self.model.head.register_forward_hook(
            lambda _module, inputs, _output: captured.append(inputs[0].detach())
        )
        try:
            with torch.inference_mode():
                rows = self.model(pieces)
        except ValueError as error:
            if "exceeds max_length" in str(error):
                raise PromptTooLongError(str(error)) from error
            raise
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("expected exactly one Open-Jev head call per batch")
        upstream = torch.cat(
            [row[1:] if piece["kind"] == "noul" else row for row, piece in zip(rows, pieces)]
        )
        with torch.inference_mode():
            replay = self.core(captured[0])
        difference = float((replay.float().cpu() - upstream.float().cpu()).abs().max())
        self.parity_max_abs_diff = max(self.parity_max_abs_diff, difference)
        if difference > PARITY_TOLERANCE:
            raise RuntimeError(f"DecPort core differs from the Open-Jev head by {difference}")
        self.input_tokens += int(getattr(self.model, "last_input_tokens", 0))
        self.candidate_sequences += len(upstream)
        hidden = captured[0].float().cpu()
        if hidden.shape[0] != len(upstream):
            raise RuntimeError("captured head inputs do not match the candidate count")
        counts = [1 if piece["kind"] == "noul" else len(row) for row, piece in zip(rows, pieces)]
        return [row.float().cpu().tolist() for row in rows], list(torch.split(hidden, counts))


def installed_openjev_commit() -> str | None:
    """The VCS commit recorded for the installed ``open-jev`` distribution, if any."""

    from importlib import metadata

    try:
        direct_url = metadata.distribution("open-jev").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def _record_length(record: dict[str, object]) -> int:
    from jev.api import candidate_prompts

    return max(len(prompt) for prompt in candidate_prompts(record))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
