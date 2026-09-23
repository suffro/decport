#!/usr/bin/env python3
"""Evaluate the final-gate source system and every selected target condition on JevBench.

Only the JevBench v1.4.0 public subset (231/534) is available, and every number is labeled that
way. Scoring, summaries, and task definitions come from a clean checkout of the pinned JevBench
commit. The adapter/module per (target, condition) is chosen by the pre-registered label-free rule
in `decport.final_gate.select_jevbench_seeds`, before any JevBench task is scored.
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
from safetensors.torch import load_file

from decport.adapter import LowRankAdapter
from decport.broad import score_core_batch
from decport.broad_experiment import TARGET_BACKBONES
from decport.decision_core import MLPDecisionCore, ScalarIdentityCore
from decport.experiment import resolve_device
from decport.final_gate import (
    CONDITIONS,
    TargetDecisionModule,
    core_calibrated_logits,
    select_jevbench_seeds,
)
from decport.jevbench import (
    JEVBENCH_COMMIT,
    PUBLIC_SCOPE,
    DecPortJevBenchAdapter,
    load_jevbench,
    run_public_subset,
)
from decport.model import DecPort
from decport.openjev import (
    OPENJEV_COMMIT,
    OpenJevTeacher,
    download_openjev_package,
    installed_openjev_commit,
    load_openjev_core,
    render_openjev_prompts,
    verify_openjev_package,
)

# The published Open-Jev JevBench protocol used a 16,384-token limit.
DEFAULT_MAX_LENGTH = 16384
SELECTION_RULE = (
    "trained conditions: the seed with the lowest final-epoch training loss for that (target, "
    "condition), ties to the lowest seed; untrained adapter: seed 0. Fixed in decision 0007 "
    "before evaluation; no JevBench answer or evaluation label is read."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jevbench-root", required=True, help=f"clean checkout of {JEVBENCH_COMMIT}"
    )
    parser.add_argument("--run", required=True, help="a completed final-gate run directory")
    parser.add_argument("--output", required=True, help="new directory outside Git")
    parser.add_argument("--package-dir", help="verified local Open-Jev package directory")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="bounded smoke run over the first N tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    upstream = load_jevbench(args.jevbench_root)
    summary = _json(run / "aggregate_summary.json")
    results = [_json(run / f"seed-{seed}" / "results.json") for seed in summary["seeds"]]
    selection = select_jevbench_seeds(results)
    _write(output / "selection.json", {"rule": SELECTION_RULE, "selection": selection})

    installed_commit = installed_openjev_commit()
    if installed_commit != OPENJEV_COMMIT:
        raise RuntimeError(f"installed Open-Jev is {installed_commit}, expected {OPENJEV_COMMIT}")
    package = verify_openjev_package(
        Path(args.package_dir) if args.package_dir else download_openjev_package()
    )
    device = resolve_device(args.device)
    core = _load_core(run / "source_core").to(device)
    if core.state_sha256() != summary["decision_core_sha256"]:
        raise ValueError("the learned core artifact differs from the run's core")
    base = {
        "scope": PUBLIC_SCOPE,
        "run": str(run),
        "learned_decision_core_sha256": core.state_sha256(),
        "packages": {
            name: metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "open-jev")
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    reports = {"source-system": _run_source_system(upstream, package, core, device, output, args)}
    _write(
        output / "source-system" / "provenance.json",
        {
            **base,
            "system": "Open-Jev 2B representation -> learned DecisionCore -> calibration",
            "openjev_package": package.provenance(),
            "openjev_head_parity_max_abs_diff": reports["source-system"].pop("parity"),
        },
    )
    _release(device)

    for target, entry in summary["targets"].items():
        target_result = results[0]["targets"][target]
        backbone = TARGET_BACKBONES[target_result["backbone_type"]].from_pretrained(
            entry["model_id"],
            max_length=args.max_length,
            allow_truncation=False,
            model_kwargs={"dtype": "auto"},
        ).to(device)
        # Never run a target past its trained context; longer tasks become JevBench 422 refusals.
        context = getattr(backbone.model.config, "max_position_embeddings", None)
        if context:
            backbone.max_length = min(backbone.max_length, int(context))
        for condition in CONDITIONS:
            seed = selection[target][condition]
            model, artifact = _load_condition(run, seed, target, condition, backbone, core, device)

            def predictor(decision, model=model):
                with torch.inference_mode():
                    logits = score_core_batch(model, [decision], render_openjev_prompts)[0]
                return torch.softmax(logits.double(), dim=0).tolist()

            name = f"{target}-{condition}-seed{seed}"
            adapter = DecPortJevBenchAdapter(
                predictor,
                name="decport_final_gate_target",
                model=f"{entry['model_id']}+{condition}:{artifact['sha256'][:16]}",
                upstream=upstream,
                runtime={"max_length": backbone.max_length, "condition": condition},
            )
            reports[name] = run_public_subset(upstream, adapter, output / name, limit=args.limit)
            _write(
                output / name / "provenance.json",
                {
                    **base,
                    "target": target,
                    "model_id": entry["model_id"],
                    "resolved_revision": getattr(backbone.model.config, "_commit_hash", None),
                    "condition": condition,
                    "selected_seed": seed,
                    "selection_rule": SELECTION_RULE,
                    "artifact": artifact,
                },
            )
        del backbone
        _release(device)

    _write(output / "summary.json", _summarize(upstream, output, reports, selection, args.limit))
    print(json.dumps(_json(output / "summary.json")["systems"], indent=2, sort_keys=True))


def _run_source_system(upstream, package, core, device, output, args):
    """The source system: frozen Open-Jev representation -> learned core -> calibration."""

    teacher_device = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
    teacher = OpenJevTeacher.from_package(
        package, load_openjev_core(package).to(device), device=teacher_device,
        max_length=args.max_length,
    )

    def predictor(decision):
        _, representations = teacher.raw_logits_and_representations(
            [decision], candidate_batch_size=1
        )
        logits = core_calibrated_logits(core, [decision], representations)[0]
        return torch.softmax(logits.double(), dim=0).tolist()

    adapter = DecPortJevBenchAdapter(
        predictor,
        name="decport_final_gate_source_system",
        model=f"openjev-2b-representation+learned-core:{core.state_sha256()[:16]}",
        upstream=upstream,
        runtime={"candidate_batch_size": 1, "max_length": args.max_length},
    )
    report = run_public_subset(upstream, adapter, output / "source-system", limit=args.limit)
    return {**report, "parity": teacher.parity_max_abs_diff}


def _load_core(directory: Path) -> MLPDecisionCore:
    config = _json(directory / "config.json")
    core = MLPDecisionCore(
        config["input_size"],
        config["temperature"],
        state=load_file(directory / "core.safetensors"),
        hidden_sizes=tuple(config["hidden_sizes"]),
    )
    if core.state_sha256() != config["state_sha256"]:
        raise ValueError(f"core artifact digest differs: {directory}")
    return core


def _load_condition(run, seed, target, condition, backbone, core, device):
    directory = run / f"seed-{seed}" / "targets" / target / condition
    config = _json(directory / "config.json")
    if config["module"] == "LowRankAdapter":
        module = LowRankAdapter(config["input_size"], config["output_size"], rank=config["rank"])
        module.load_state_dict(load_file(directory / "adapter.safetensors"))
        head = core
        if condition == "random_core_distillation":
            head = _load_core(run / f"seed-{seed}" / "random_core").to(device)
        filename = "adapter.safetensors"
    else:
        module = TargetDecisionModule(config["input_size"], tuple(config["hidden_sizes"]))
        module.load_state_dict(load_file(directory / "module.safetensors"))
        head = ScalarIdentityCore(config["decision_core_temperature"])
        filename = "module.safetensors"
    if head.state_sha256() != config["decision_core_sha256"]:
        raise ValueError(f"{directory} was not trained for this decision core")
    model = DecPort(backbone, module.to(device), head.to(device))
    model.eval()
    artifact = {
        "path": str(directory),
        "sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
        "decision_core_sha256": config["decision_core_sha256"],
    }
    return model, artifact


def _summarize(upstream, output, reports, selection, limit):
    tasks = list(upstream.tasks[:limit] if limit else upstream.tasks)
    predicted = {
        name: {
            record["task_id"]: record["predicted"] if record["ok"] else None
            for record in map(json.loads, (output / name / "per_item.jsonl").read_text(
                encoding="utf-8").splitlines())
        }
        for name in reports
    }
    source = predicted["source-system"]
    systems = {}
    for name, report in reports.items():
        summary = report["summary"]
        systems[name] = {
            "n_correct": summary["n_correct"],
            "n_planned": summary["n_planned"],
            "n_valid": summary["n_valid"],
            "accuracy": summary["accuracy"],
            "brier_mean": summary["brier_mean"],
            "ece_top_label": summary["ece"]["ece"],
            "by_question_type": {
                kind: [value["n_correct"], value["n_planned"]]
                for kind, value in report["by_question_type"].items()
            },
            "by_tier": {
                tier: [value["n_correct"], value["n_planned"]]
                for tier, value in report["by_tier"].items()
            },
            "failures": report["failures_by_status_and_kind"],
            "per_item_results_sha256": report["raw_artifacts"]["per_item_results_sha256"],
        }
        if name != "source-system":
            systems[name]["source_system_top_choice_agreement"] = sum(
                label is not None and label == source.get(task_id)
                for task_id, label in predicted[name].items()
            )
    return {
        "scope": PUBLIC_SCOPE,
        "complete_public_subset": limit is None,
        "selection_rule": SELECTION_RULE,
        "selection": selection,
        "uniform_guess_expected_correct": sum(1 / len(task.labels) for task in tasks),
        "systems": systems,
    }


def _release(device):
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
