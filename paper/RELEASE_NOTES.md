# DecPort research article — 2026-09-24

**Proposed tag:** `paper-2026-09-24`<br>
**Proposed release title:** `DecPort research article — 2026-09-24`<br>
**DOI:** [10.5281/zenodo.22947927](https://doi.org/10.5281/zenodo.22947927)

This is a research-artifact and paper release. It is **not** a DecPort v0.1 software release; the
software ship gate returned NO-SHIP and no v0.1 software release was made.

## Release description

This release accompanies the technical research article **Distinguishing Cross-Backbone
Decision-Behavior Transfer from Frozen-Core Portability** by Lorenzo Suffritti, dated 24 September
2026.

The final experimental verdict is **NO-SHIP** for the original hypothesis that one learned frozen
`DecisionCore` provides useful reusable decision computation across the tested heterogeneous frozen
backbones through lightweight adapters. In the preregistered nonlinear final gate, the learned core
did not satisfy the primary criterion against a scale-matched random nonlinear core and did not
satisfy the practical-utility criterion against target-specific distillation.

A separate finding remains supported within the tested setting: label-free, input-specific
decision-behavior distillation transferred source behavior into SmolLM2, Gemma 3, and TinyLlama.
Choice and Score carry most of that result. Noul did not recover input-specific source behavior;
OOD effects and calibration are weaker.

Accepted evidence is archived in the repository:

- [Decision 0006 accepted Open-Jev archive](../benchmarks/accepted/openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/)
- [Decision 0007 final report](../benchmarks/accepted/decport-final-gate-v0.1/FINAL_REPORT.md)
- [Decision 0007 accepted archive](../benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/)
- [Technical research article](decport-technical-research-article.pdf)

The accepted runs passed their audits. Their same-environment reproductions were bit-exact.
Independent and cross-hardware replication were not performed.

Research conducted with the support of artificial intelligence (AI).
