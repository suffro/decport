# DecPort

DecPort tested whether input-specific probabilistic decision behavior can be transferred across
heterogeneous frozen language-model backbones without target labels, and whether a particular
learned frozen `DecisionCore` contributes reusable functionality after the backbone change.

> **Final scientific outcome: NO-SHIP for the original reusable learned-core hypothesis.**
>
> The preregistered nonlinear final gate did not support reusable learned `DecisionCore`
> portability in the tested setting. Separately, label-free input-specific decision-behavior
> distillation remains supported within that setting. Choice and Score carry most of the positive
> transfer result; Noul did not recover input-specific source behavior in either accepted stage.
> OOD effects are smaller and calibration is weaker.

## Paper

**[Distinguishing Cross-Backbone Decision-Behavior Transfer from Frozen-Core
Portability](paper/decport-technical-research-article.pdf)**

Lorenzo Suffritti<br>
Technical research article, 24 September 2026

DOI: [10.5281/zenodo.22947927](https://doi.org/10.5281/zenodo.22947927)

Research conducted with the support of artificial intelligence (AI).

The article covers the full experimental progression through decision 0007. Citation metadata is
available in [`CITATION.cff`](CITATION.cff), and the paper-specific overview is in
[`paper/README.md`](paper/README.md).

## Evidence at a glance

| Question | Result |
| --- | --- |
| Input-specific cross-backbone behavior transfer (RQ1) | Supported within the tested setting |
| Reusable learned `DecisionCore` portability (RQ2) | Not supported |
| Final preregistered ship gate | **NO-SHIP** |
| Audit | Passed |
| Same-environment reproduction | Bit-exact |
| Independent or cross-hardware replication | Not performed |

RQ1 compares correct-teacher transfer with a decision-type- and width-matched mismatched teacher.
RQ2 asks the stronger question: whether the particular learned frozen core contributes reusable
functionality beyond a matched arbitrary frozen core under the same teacher signal. Evidence for
RQ1 does not imply evidence for RQ2.

## Accepted evidence

### Decision 0006 — Open-Jev linear `DecisionCore`

The first accepted stage used the real released Open-Jev 2B source system, its scalar
`Linear(2048 -> 1)` core, five seeds, and three heterogeneous frozen targets. Label-free behavior
transfer was positive overall and was carried by Choice and Score. Noul failed. The matched random
core comparison also exposed the linear-core identifiability problem: the experiment provided no
evidence that the particular pretrained rank-one core mattered.

- [Decision record](.context/decisions/0006-openjev-decision-core-transfer.md)
- [Accepted series](benchmarks/accepted/openjev-transfer-v0.1/)
- [Accepted five-seed archive](benchmarks/accepted/openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/)

### Decision 0007 — nonlinear final ship gate

The preregistered final gate used a learned nonlinear 1,118,977-parameter source `DecisionCore`, a
weak rank-128 target adapter, a scale-matched random nonlinear core, mismatched-teacher and
untrained controls, and a target-specific distilled baseline. Criteria A–E tested whether the
learned core mattered, transfer was input-specific, effects survived distribution shift, external
behavior cleared the JevBench public-subset threshold, and the shared-core route had practical
utility. Criteria A, D, and E failed, producing **NO-SHIP**. The audit passed and a second execution
in the same hardware/software environment was bit-exact.

- [Decision record](.context/decisions/0007-decport-final-ship-gate.md)
- [Final report](benchmarks/accepted/decport-final-gate-v0.1/FINAL_REPORT.md)
- [Accepted five-seed archive](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/)
- [Mechanical verdict](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/verdict.json)
- [Audit](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/audit.json)
- [Reproduction comparison](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/reproduction/reproduction_check.json)
- [Run provenance](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/run/provenance.json)

The run and reproduction provenance both record the frozen protocol commit
`d0aa96f4defc3ce202996be1f75d669b791d57fc` and `git_worktree_dirty:true`. The accepted archive
verifies the frozen repository configuration and data hashes, but it is not described as a
clean-checkout guarantee.

## Research and evidence map

[`RESEARCH.md`](RESEARCH.md) maps decisions 0001–0007 and makes the progression from supervised
transfer to the final falsification gate explicit. The repository preserves three evidence levels:

- [`benchmarks/pilots/`](benchmarks/pilots/) — bounded pipeline checks, not benchmark evidence;
- [`benchmarks/diagnostics/`](benchmarks/diagnostics/) — controlled studies used to refine the
  question, not accepted benchmark claims;
- [`benchmarks/accepted/`](benchmarks/accepted/) — meaningful-scale multi-seed runs that passed
  audit and reproduction before acceptance.

Acceptance certifies the archived measurement, including its negative findings; it does not extend
the claim beyond the recorded models, datasets, tasks, and training budget.

## Inspect and reproduce

Install the development and Open-Jev dependencies and run the ordinary package checks:

```bash
uv sync --extra dev --extra openjev
uv run ruff check .
uv run pytest
```

Real-model tests are opt-in because they download checkpoints:

```bash
DECPORT_RUN_REAL_MODELS=1 uv run pytest tests/test_real_backbones.py tests/test_real_openjev.py
```

The final report and verdict can be inspected without rerunning the expensive experiment. The
[accepted final-gate archive README](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/README.md)
records the exact data preparation, full run, audit, comparison, JevBench public-subset, and
mechanical-verdict commands. The archived evidence includes configs, data manifests, per-seed
results, aggregate tables, artifact hashes, audit failure-path probes, and reproduction results.
Base-model checkpoints and downloaded datasets are intentionally not committed.

The v0.1 Python package remains a research implementation. Its basic API exposes frozen backbone
wrappers, adapters, typed Choice/Noul/Score decision cores, experiment runners, and audit scripts.
No DecPort v0.1 software release was made.

## Scope and provenance

The conclusion is bounded to three target families, one Open-Jev/Qwen source representation
boundary, the archived datasets and budgets, one nonlinear core design, and the preregistered
criteria. It does not prove mathematical equivalence between learned and random cores, and it does
not imply that nothing transferred.

Historical evidence remains in Git so the commit SHAs cited by the article stay valid. Future large
publication artifacts should preferably use release assets or an archival deposition rather than
rewriting existing provenance.

The code is licensed under the [Apache License 2.0](LICENSE).
