#!/usr/bin/env python3
"""Evaluate the Open-Jev teacher or a DecPort-ported target on the public JevBench subset.

Scoring, summaries, and task definitions come from a clean checkout of the pinned JevBench commit.
Results cover only the 231 public tasks, never the full 534-task benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch

from decport.broad import score_core_batch
from decport.broad_experiment import TARGET_BACKBONES
from decport.experiment import resolve_device
from decport.jevbench import (
    JEVBENCH_COMMIT,
    DecPortJevBenchAdapter,
    load_jevbench,
    run_public_subset,
)
from decport.openjev import (
    OPENJEV_2B_REPO_ID,
    OPENJEV_2B_REVISION,
    OPENJEV_COMMIT,
    OpenJevTeacher,
    download_openjev_package,
    installed_openjev_commit,
    load_openjev_core,
    render_openjev_prompts,
    verify_openjev_package,
)
from decport.serialization import load_artifact

# The published Open-Jev JevBench protocol used a 16,384-token limit.
DEFAULT_MAX_LENGTH = 16384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jevbench-root", required=True, help=f"clean checkout of {JEVBENCH_COMMIT}"
    )
    parser.add_argument("--output", required=True, help="new run directory outside Git")
    parser.add_argument("--system", choices=("openjev-teacher", "decport-target"), required=True)
    parser.add_argument("--package-dir", help="verified local Open-Jev package directory")
    parser.add_argument("--artifact", help="DecPort adapter artifact for --system decport-target")
    parser.add_argument("--backbone-type", choices=sorted(TARGET_BACKBONES))
    parser.add_argument("--model-id", help="target backbone model ID")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="bounded smoke run over the first N tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream = load_jevbench(args.jevbench_root)
    installed_commit = installed_openjev_commit()
    if installed_commit != OPENJEV_COMMIT:
        raise RuntimeError(f"installed Open-Jev is {installed_commit}, expected {OPENJEV_COMMIT}")
    package = verify_openjev_package(
        Path(args.package_dir) if args.package_dir else download_openjev_package()
    )
    device = resolve_device(args.device)
    core = load_openjev_core(package).to(device)
    provenance: dict[str, object] = {
        "openjev": {
            "pinned_commit": OPENJEV_COMMIT,
            "installed_commit": installed_commit,
            "model_repository": OPENJEV_2B_REPO_ID,
            "model_revision": OPENJEV_2B_REVISION,
            "package": package.provenance(),
            "decision_core_sha256": core.state_sha256(),
        },
        "packages": {
            name: metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "open-jev")
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    if args.system == "openjev-teacher":
        teacher_device = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
        teacher = OpenJevTeacher.from_package(
            package, core, device=teacher_device, max_length=args.max_length
        )
        predictor = teacher.calibrated_probabilities
        name, model_name = "decport_openjev_teacher", f"{OPENJEV_2B_REPO_ID}@{OPENJEV_2B_REVISION}"
        runtime = {"candidate_batch_size": 1, "max_length": args.max_length, "prefix_cache": False}
    else:
        if not (args.artifact and args.backbone_type and args.model_id):
            raise SystemExit("--artifact, --backbone-type, and --model-id are required")
        artifact = Path(args.artifact)
        artifact_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
        if artifact_config["metadata"].get("decision_core_sha256") != core.state_sha256():
            raise ValueError("the adapter artifact was not trained for this Open-Jev core")
        backbone = TARGET_BACKBONES[args.backbone_type].from_pretrained(
            args.model_id,
            max_length=args.max_length,
            allow_truncation=False,
            model_kwargs={"dtype": "auto"},
        ).to(device)
        # Never run a target past its trained context; longer tasks become JevBench 422 refusals.
        context = getattr(backbone.model.config, "max_position_embeddings", None)
        if context:
            backbone.max_length = min(backbone.max_length, int(context))
        model = load_artifact(artifact, backbone, head=core)
        model.adapter.to(device)
        model.eval()

        def predictor(decision):
            with torch.inference_mode():
                logits = score_core_batch(model, [decision], render_openjev_prompts)[0]
            return torch.softmax(logits.double(), dim=0).tolist()

        adapter_sha256 = hashlib.sha256((artifact / "adapter.safetensors").read_bytes()).hexdigest()
        name = "decport_ported_target"
        model_name = f"{args.model_id}+decport-adapter:{adapter_sha256[:16]}+openjev-2b-core"
        runtime = {"max_length": backbone.max_length, "backbone_type": args.backbone_type}
        provenance["target"] = {
            "model_id": args.model_id,
            "resolved_revision": getattr(backbone.model.config, "_commit_hash", None),
            "artifact": str(artifact),
            "artifact_role": artifact_config["metadata"].get("role"),
            "adapter_sha256": adapter_sha256,
        }

    adapter = DecPortJevBenchAdapter(
        predictor, name=name, model=model_name, upstream=upstream, runtime=runtime
    )
    report = run_public_subset(upstream, adapter, args.output, limit=args.limit)
    if args.system == "openjev-teacher":
        provenance["teacher_run"] = {
            "core_parity_max_abs_diff": teacher.parity_max_abs_diff,
            "candidate_sequences": teacher.candidate_sequences,
            "input_tokens": teacher.input_tokens,
        }
    output = Path(args.output)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "n_correct": summary["n_correct"],
                "n_planned": summary["n_planned"],
                "accuracy": summary["accuracy"],
                "brier_mean": summary["brier_mean"],
                "by_question_type": {
                    kind: {"n_correct": value["n_correct"], "n_planned": value["n_planned"]}
                    for kind, value in report["by_question_type"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
