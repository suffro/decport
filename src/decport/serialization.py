"""Safe, explicit serialization for DecPort adapters and heads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from safetensors.torch import load_file, save_file

from decport.adapter import BackboneAdapter
from decport.backbones.base import DecPortBackbone
from decport.head import DecisionHead
from decport.model import DecPort

FORMAT_VERSION = 1


def save_artifact(
    model: DecPort,
    directory: str | Path,
    *,
    include_head: bool,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save an adapter and, optionally, its shared/native head."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": FORMAT_VERSION,
        "backbone_model_id": getattr(model.backbone, "model_id", None),
        "adapter_input_size": model.adapter.input_size,
        "shared_size": model.adapter.shared_size,
        "includes_head": include_head,
        "metadata": dict(metadata or {}),
    }
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(model.adapter.state_dict(), destination / "adapter.safetensors")
    if include_head:
        save_file(model.head.state_dict(), destination / "head.safetensors")
    return destination


def load_artifact(
    directory: str | Path,
    backbone: DecPortBackbone,
    *,
    require_head: bool = True,
    head: DecisionHead | None = None,
) -> DecPort:
    """Load local DecPort components against an already-instantiated backbone."""

    source = Path(directory)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported artifact format version: {config.get('format_version')}")
    input_size = config.get("adapter_input_size")
    shared_size = config.get("shared_size")
    if not isinstance(input_size, int) or not isinstance(shared_size, int):
        raise ValueError("artifact dimensions are missing or invalid")
    if input_size != backbone.hidden_size:
        raise ValueError(
            f"artifact expects hidden size {input_size}, backbone exposes {backbone.hidden_size}"
        )

    adapter = BackboneAdapter(input_size=input_size, shared_size=shared_size)
    adapter.load_state_dict(load_file(source / "adapter.safetensors"))

    includes_head = config.get("includes_head") is True
    if head is None:
        if require_head and not includes_head:
            raise ValueError("artifact does not include a decision head")
        head = DecisionHead(shared_size=shared_size)
        if includes_head:
            head.load_state_dict(load_file(source / "head.safetensors"))
    elif head.shared_size != shared_size:
        raise ValueError("provided head does not match the artifact shared size")

    return DecPort(backbone=backbone, adapter=adapter, head=head)
