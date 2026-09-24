# Distinguishing Cross-Backbone Decision-Behavior Transfer from Frozen-Core Portability

**Author:** Lorenzo Suffritti<br>
**Type:** Technical research article<br>
**Date:** 24 September 2026<br>
**DOI:** [10.5281/zenodo.22947927](https://doi.org/10.5281/zenodo.22947927)

Research conducted with the support of artificial intelligence (AI).

[Read the paper](decport-technical-research-article.pdf).

## Summary

The article reconstructs the complete DecPort experimental progression through decision 0007. It
separates two questions that earlier prototype framing could conflate:

- **RQ1 — behavior transfer:** whether input-specific probabilistic decision behavior can move from
  a source system to heterogeneous frozen target backbones without target ground-truth labels;
- **RQ2 — `DecisionCore` portability:** whether the particular learned frozen core contributes
  reusable functionality beyond a matched arbitrary frozen core under the same teacher signal.

The accepted evidence supports RQ1 within the tested backbones, datasets, tasks, and training
budget. Choice and Score carry most of the positive result; Noul does not recover input-specific
source behavior, OOD effects are smaller, and calibration is weaker.

RQ2 is not supported. The preregistered nonlinear final gate found near-zero or negative
learned-core gains over the scale-matched random core, and the shared-core route did not satisfy the
preregistered practical-utility criterion against target-specific distillation. The final verdict
is **NO-SHIP** for the original reusable learned-core hypothesis. This does not mean that nothing
transferred.

## Evidence and provenance

The research progression is mapped in [`../RESEARCH.md`](../RESEARCH.md), with chronological
records under [`../.context/decisions/`](../.context/decisions/). Accepted evidence is indexed in
[`../benchmarks/accepted/`](../benchmarks/accepted/).

### Decision 0006 — accepted Open-Jev linear-core evidence

- Accepted evidence commit: [`51f221348163b320e46111d0857a4a255e304ec0`](https://github.com/suffro/decport/tree/51f221348163b320e46111d0857a4a255e304ec0)
- Implementation provenance: [`0ab715530a3facf5d95cba9bedf6949f82ccdd5e`](https://github.com/suffro/decport/tree/0ab715530a3facf5d95cba9bedf6949f82ccdd5e)
- [Decision record](../.context/decisions/0006-openjev-decision-core-transfer.md)
- [Accepted archive](../benchmarks/accepted/openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/)

Decision 0006 established reproducible input-specific Choice/Score behavior transfer from the
released Open-Jev source system, while the random-core comparison exposed the scalar linear core's
identifiability problem. Noul failed.

### Decision 0007 — accepted nonlinear final gate

- Preregistered protocol and implementation: [`d0aa96f4defc3ce202996be1f75d669b791d57fc`](https://github.com/suffro/decport/tree/d0aa96f4defc3ce202996be1f75d669b791d57fc)
- Accepted result and archive: [`cf9939e213ca2c045bc6bf6968b68bcc8f8656be`](https://github.com/suffro/decport/tree/cf9939e213ca2c045bc6bf6968b68bcc8f8656be)
- [Decision record](../.context/decisions/0007-decport-final-ship-gate.md)
- [Final report](../benchmarks/accepted/decport-final-gate-v0.1/FINAL_REPORT.md)
- [Accepted archive](../benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/)
- [Mechanical verdict](../benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/verdict.json)
- [Audit](../benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/audit.json)
- [Reproduction comparison](../benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/reproduction/reproduction_check.json)

The final-gate audit passed, and a second execution was bit-exact in the same hardware/software
environment. Independent and cross-hardware replication were not performed. Both the run and
reproduction provenance record the protocol commit above together with `git_worktree_dirty:true`;
the commit is therefore a recorded base/protocol revision, not a clean-checkout guarantee.

## Citation

The preferred citation is the article rather than the Python package. Machine-readable citation
metadata is available in [`../CITATION.cff`](../CITATION.cff). The canonical published record is
available through [Zenodo](https://doi.org/10.5281/zenodo.22947927).

Publication release notes are prepared in [`RELEASE_NOTES.md`](RELEASE_NOTES.md). No DecPort v0.1
software release was made.
