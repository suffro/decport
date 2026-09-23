# DecPort final ship gate v0.1 — accepted runs

This series is the final pre-registered test of the DecPort hypothesis (decision 0007). It asks
whether one learned nonlinear DecisionCore, trained once on the source side and frozen, is a useful
reusable core across heterogeneous frozen backbones reached through deliberately weak rank-128
adapters. The controls are a scale-matched random core, a mismatched teacher, an untrained adapter,
and a target-specific distilled module.

**FINAL VERDICT: NO-SHIP.** See [`FINAL_REPORT.md`](FINAL_REPORT.md).

| Date | Platform | Seeds | Source train / calibration | Transfer train / ID / OOD | Status |
|---|---|---:|---:|---:|---|
| 2026-09-23 | [Windows + WSL2, RTX 4060 Ti](2026-09-23-wsl2-rtx4060ti-seeds0-4/) | 0–4 | 4,500 / 900 | 4,500 / 2,000 / 1,500 | Accepted; reproduced bit-exactly; NO-SHIP |
