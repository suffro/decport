# Accepted benchmark evidence

Runs here are at meaningful scale, use multiple seeds, passed their audit, and were reproduced
before acceptance. Acceptance certifies the measurement, including its negative findings. It
does not turn a result into a claim beyond what the run's report states.

| Series | Run | Reproduction | Main result |
|---|---|---|---|
| Open-Jev DecisionCore transfer v0.1 | [`2026-09-23-wsl2-rtx4060ti-seeds0-4`](openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/) | Bit-exact (same hardware) | Open-Jev behavior transfers on Choice and Score to all three targets (+0.18 ID / +0.07 OOD over the mismatched teacher, 5/5 seeds). No stable learned-core advantage over the matched random core was demonstrated. Noul does not transfer. |
| **DecPort final ship gate v0.1** | [`2026-09-23-wsl2-rtx4060ti-seeds0-4`](decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/) | Bit-exact (same hardware) | **FINAL VERDICT: NO-SHIP** ([report](decport-final-gate-v0.1/FINAL_REPORT.md)). The learned nonlinear core did not meet the preregistered advantage threshold over the scale-matched random core (A − B −0.015 to +0.015 ID) and did not beat a target-specific distilled module. Label-free behavior transfer itself is input-specific on all three targets. |
