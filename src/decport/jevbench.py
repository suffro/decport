"""Thin DecPort adapter for the pinned public JevBench harness.

JevBench stays an external checkout. DecPort imports JevBench's own task loader, request builder,
runner, scoring, and summaries from a clean checkout of one pinned commit, and never copies,
edits, or reconstructs its tasks. Only the public subset (231 of 534 tasks) is available, so every
result is labeled as a public-subset result. Raw per-item records stay in the run directory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from decport.backbones.base import PromptTooLongError
from decport.decision_core import TypedDecision

JEVBENCH_REPOSITORY = "https://github.com/fstandhartinger/jevbench"
JEVBENCH_COMMIT = "2fa63fa3226cb369795525ed011800f57dcbd894"
JEVBENCH_RELEASE = "v1.4.0"
PUBLIC_TIERS = {
    "original": (72, "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "easy": (48, "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "hard": (111, "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
PUBLIC_SCOPE = (
    "JevBench public subset only: 231 of 534 tasks (72 original, 48 easy, 111 hard). The private "
    "and judge tiers are unavailable; this is not a full-benchmark result."
)
# JevBench's runner treats 422 as the system refusing one input (for example, over its context
# limit): the task is scored wrong, but it does not count toward the infrastructure stop rule.
CONTEXT_LIMIT_STATUS = 422

Predictor = Callable[[TypedDecision], Sequence[float]]


@dataclass(frozen=True)
class JevBenchUpstream:
    """Modules and public tasks imported from a verified JevBench checkout."""

    root: Path
    commit: str
    tasks: tuple
    tier_by_task: dict[str, str]
    manifest: dict[str, object]
    task_module: ModuleType
    scoring: ModuleType
    summarize: ModuleType
    base: ModuleType
    runner: ModuleType
    budget: ModuleType


def load_jevbench(root: str | Path, *, expected_commit: str = JEVBENCH_COMMIT) -> JevBenchUpstream:
    """Import JevBench only from a clean checkout of the pinned commit and verify public data."""

    root = Path(root).resolve()
    commit = _git(root, "rev-parse", "HEAD").strip()
    if commit != expected_commit:
        raise ValueError(f"JevBench checkout is at {commit}, expected {expected_commit}")
    if _git(root, "status", "--porcelain").strip():
        raise ValueError("JevBench checkout has local modifications")
    for name, module in list(sys.modules.items()):
        if name == "jevbench" or name.startswith("jevbench."):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise ValueError("a different JevBench source is already imported")
    sys.path.insert(0, str(root))
    try:
        modules = {
            name: importlib.import_module(f"jevbench.{name}")
            for name in ("tasks", "scoring", "summarize", "adapters.base", "runner", "budget")
        }
    finally:
        sys.path.remove(str(root))
    tasks = []
    tier_by_task = {}
    manifest: dict[str, object] = {}
    for tier, (count, expected_sha256) in PUBLIC_TIERS.items():
        source = root / "datasets" / "public" / f"{tier}.jsonl"
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"public JevBench {tier} file checksum differs")
        tier_tasks = modules["tasks"].load_jsonl(str(source))
        if len(tier_tasks) != count or any(task.split != "public" for task in tier_tasks):
            raise ValueError(f"public JevBench {tier} task count or split differs")
        tasks.extend(tier_tasks)
        tier_by_task.update({task.id: tier for task in tier_tasks})
        manifest[tier] = {
            "rows": count,
            "file_sha256": expected_sha256,
            "canonical_sha256": modules["tasks"].dataset_hash(tier_tasks),
            "question_types": dict(Counter(task.question["type"] for task in tier_tasks)),
        }
    if len(tier_by_task) != len(tasks):
        raise ValueError("duplicate JevBench task IDs")
    return JevBenchUpstream(
        root=root,
        commit=commit,
        tasks=tuple(tasks),
        tier_by_task=tier_by_task,
        manifest=manifest,
        task_module=modules["tasks"],
        scoring=modules["scoring"],
        summarize=modules["summarize"],
        base=modules["adapters.base"],
        runner=modules["runner"],
        budget=modules["budget"],
    )


def typed_decision_from_task(task, build_question: Callable[[object], dict]) -> TypedDecision:
    """Convert a JevBench task with JevBench's own request builder; the gold label is never read."""

    question = build_question(task)
    return TypedDecision(
        state=task.state,
        kind=question["type"],
        instructions=question["instructions"],
        criteria=question.get("criteria"),
        dataset="jevbench_public",
        task_family=task.family,
        decision_id=task.id,
    )


def jevbench_probabilities(
    decision: TypedDecision,
    probabilities: Sequence[float],
    labels: Sequence[str],
) -> dict[str, float]:
    """Map a typed distribution onto JevBench's exact label set, without renormalizing."""

    if len(probabilities) != len(decision.options):
        raise ValueError("probability count differs from the decision's answer keys")
    if decision.kind == "noul":
        mapped = {"no": float(probabilities[0]), "yes": float(probabilities[1])}
    else:
        mapped = {key: float(value) for key, value in zip(decision.options, probabilities)}
    if set(mapped) != set(labels):
        raise ValueError("DecPort answer keys differ from the JevBench label set")
    return mapped


class DecPortJevBenchAdapter:
    """JevBench adapter over any DecPort typed-decision predictor (teacher or ported target)."""

    cost_basis = "local_gpu_no_provider_tariff"
    price_input_per_m = None
    price_output_per_m = None

    def __init__(
        self,
        predictor: Predictor,
        *,
        name: str,
        model: str,
        upstream: JevBenchUpstream,
        runtime: dict[str, object] | None = None,
    ) -> None:
        self.predictor = predictor
        self.name = name
        self.model = model
        self.upstream = upstream
        self.runtime = dict(runtime or {})

    def reserve_estimate(self, task) -> float:
        return 0.0

    def run(self, task):
        result = self.upstream.base.DecisionResult(
            adapter=self.name, ok=False, probs_source="native", model=self.model
        )
        decision = typed_decision_from_task(task, self.upstream.base.build_question)
        result.request_body = {"state": task.state, "questions": {"decision": decision.question()}}
        started = time.perf_counter()
        try:
            probabilities = [float(value) for value in self.predictor(decision)]
            result.probs = jevbench_probabilities(decision, probabilities, task.labels)
        except PromptTooLongError as error:
            result.latency_s = time.perf_counter() - started
            result.status = CONTEXT_LIMIT_STATUS
            result.error = f"context limit: {str(error)[:250]}"
            return result
        except Exception as error:  # noqa: BLE001 - a failed decision is a failed attempt
            result.latency_s = time.perf_counter() - started
            result.error = f"{type(error).__name__}: {str(error)[:250]}"
            return result
        result.latency_s = time.perf_counter() - started
        result.raw = {"probabilities": probabilities, "runtime": self.runtime}
        result.ok = True
        return result


def run_public_subset(
    upstream: JevBenchUpstream,
    adapter: DecPortJevBenchAdapter,
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Run JevBench's own runner and summaries; return a public-safe aggregate report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tasks = list(upstream.tasks[:limit] if limit else upstream.tasks)
    ledger = upstream.budget.Ledger(str(output_dir / "ledger.jsonl"), cap_usd=0.0)
    runner = upstream.runner.Runner(
        adapter, ledger, raw_dir=output_dir / "raw", default_reserve_usd=0.0
    )
    per_item = output_dir / "per_item.jsonl"
    records = runner.run_all(tasks, results_path=per_item)
    metric = upstream.summarize.metric
    summary = upstream.summarize.summarize(tasks, records)
    report = {
        "scope": PUBLIC_SCOPE,
        "complete_public_subset": len(tasks) == len(upstream.tasks),
        "upstream": {
            "repository": JEVBENCH_REPOSITORY,
            "commit": upstream.commit,
            "release": JEVBENCH_RELEASE,
            "public_files": upstream.manifest,
            "dataset_hash": upstream.task_module.dataset_hash(tasks),
        },
        "adapter": {"name": adapter.name, "model": adapter.model, "runtime": adapter.runtime},
        "summary": upstream.summarize.public_export(summary, tasks, records),
        "by_question_type": {
            kind: metric([task for task in tasks if task.question["type"] == kind], records)
            for kind in sorted({task.question["type"] for task in tasks})
        },
        "by_tier": {
            tier: metric(
                [task for task in tasks if upstream.tier_by_task[task.id] == tier], records
            )
            for tier in PUBLIC_TIERS
            if any(upstream.tier_by_task[task.id] == tier for task in tasks)
        },
        "failures_by_status_and_kind": dict(
            Counter(
                f"{record['status_code']}:{(record['error'] or 'unknown').split(':', 1)[0]}"
                for record in records
                if not record["ok"]
            )
        ),
        "raw_artifacts": {
            "per_item_results_sha256": hashlib.sha256(per_item.read_bytes()).hexdigest(),
            "per_item_results_records": len(records),
            "note": "Per-item records and raw request/response files stay in the run directory; "
            "they contain JevBench task content and are not copied into the repository.",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)
