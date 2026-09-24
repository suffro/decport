# Research and evidence map

DecPort investigated two related but distinct questions:

- **RQ1 — input-specific behavior transfer:** can probabilistic decision behavior be transferred
  from a source decision system to a heterogeneous frozen target backbone without target
  ground-truth labels?
- **RQ2 — `DecisionCore` portability:** does the particular learned frozen `DecisionCore`
  contribute reusable functionality beyond a matched arbitrary frozen core when the target adapter
  receives the same teacher signal?

RQ1 is tested by correct-teacher versus mismatched-teacher transfer. RQ2 is tested by learned-core
versus matched-random-core transfer. A positive RQ1 result does not establish RQ2.

The chronological research records are preserved in [`.context/decisions/`](.context/decisions/).
They are not rewritten here; this page maps each decision to its methodological role and evidence.

## Decision progression

| Decision | Short name | Role / status | Main methodological purpose | Evidence | Resulting implication |
| --- | --- | --- | --- | --- | --- |
| [0001](.context/decisions/0001-shared-latent-head.md) | Supervised shared-head transfer | Diagnostic foundation | Train a source adapter/head, freeze the head, and compare target transfer with a matched random frozen head under target-label supervision. | [Five-seed convergence diagnostic](benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-seeds0-4/) | Target labels let an adapter train around an arbitrary fixed head; supervised success cannot identify portable head-specific information. |
| [0002](.context/decisions/0002-label-free-latent-alignment.md) | Label-free latent alignment | Diagnostic | Remove answers and train only the target adapter to match source latent vectors. | [Seed-0 alignment diagnostic](benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-label-free-alignment-seed0/) | Alignment improved over an unaligned adapter but remained insufficient evidence for portability. |
| [0003](.context/decisions/0003-label-free-alignment-methods.md) | Alternative alignment methods | Diagnostic | Compare cosine-plus-MSE with whitening, ridge regression, and orthogonal Procrustes. | [Method diagnostic](benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-label-free-methods-seed0/) | Lower latent reconstruction loss did not reliably imply better frozen-head decisions. |
| [0004](.context/decisions/0004-label-free-decision-distillation.md) | Decision-space distillation | Diagnostic | Match the frozen source system's candidate distributions on unlabeled inputs and use a permuted-teacher control. | [Three-seed distillation diagnostic](benchmarks/diagnostics/v0.1/2026-09-22-wsl2-rtx4060ti-decision-distillation-seeds0-2/) | Bounded positive evidence emerged for input-specific behavior transfer, with weak OOD effects and worse calibration. |
| [0005](.context/decisions/0005-scaled-cross-backbone-decision-transfer.md) | Scaled cross-backbone transfer | Diagnostic | Scale data tenfold, use five seeds and two target families, and retain matched negative controls. | [Scaled diagnostic](benchmarks/diagnostics/v0.1/2026-09-23-wsl2-rtx4060ti-decision-transfer-scale-seeds0-4/) | RQ1 strengthened across targets, but the source head was still an internally trained scalar head rather than an external released system. |
| [0006](.context/decisions/0006-openjev-decision-core-transfer.md) | Open-Jev linear `DecisionCore` | Accepted evidence | Test a real released Open-Jev source system over five seeds and three targets with a matched random-core control. | [Accepted archive](benchmarks/accepted/openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/) | Choice/Score behavior transferred, Noul failed, and the random-core result exposed the linear-core identifiability confound. |
| [0007](.context/decisions/0007-decport-final-ship-gate.md) | Nonlinear final ship gate | Accepted final evidence — **NO-SHIP** | Test a learned nonlinear core through weak rank-128 adapters against scale-matched random, mismatched, untrained, and target-specific controls under preregistered criteria. | [Final report](benchmarks/accepted/decport-final-gate-v0.1/FINAL_REPORT.md) and [accepted archive](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/) | RQ1 remained supported within scope; RQ2 was not supported. Criteria A, D, and E failed, so the mechanical verdict was NO-SHIP. |

In methodological order, the progression was:

```text
supervised transfer
  -> target-label confound
  -> label-free latent alignment
  -> latent/decision mismatch
  -> decision-space distillation
  -> scaled behavior transfer
  -> real Open-Jev linear-core test
  -> identifiability confound
  -> preregistered nonlinear final gate
  -> NO-SHIP for learned-core portability
```

## Accepted evidence

The accepted evidence index is [`benchmarks/accepted/README.md`](benchmarks/accepted/README.md).
Decision 0006 is the first accepted test against a real released source system. Decision 0007 is
the final preregistered test of the original reusable learned-core hypothesis.

For decision 0007, the mechanical result and integrity evidence are directly available as:

- [`verdict.json`](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/verdict.json);
- [`audit.json`](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/audit.json);
- [`reproduction_check.json`](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/reproduction/reproduction_check.json);
- [run provenance](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/run/provenance.json) and [reproduction provenance](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/reproduction/provenance.json).

The audit passed. The second execution was bit-exact in the same hardware/software environment; it
was not an independent or cross-hardware replication. Both provenance files record
`git_worktree_dirty:true`, so the recorded protocol commit is not presented as a clean-checkout
guarantee.

## Evidence levels

The hierarchy in [`benchmarks/`](benchmarks/) is deliberate:

- [`pilots/`](benchmarks/pilots/) validates bounded execution paths;
- [`diagnostics/`](benchmarks/diagnostics/) records controlled methodological studies;
- [`accepted/`](benchmarks/accepted/) contains meaningful-scale multi-seed runs accepted only after
  audit and reproduction.

Pilot and diagnostic evidence remains part of the research chronology, but neither is promoted to
an accepted benchmark claim. Existing evidence stays in Git to preserve the historical commit SHAs
cited by the paper.
