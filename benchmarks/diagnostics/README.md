# Diagnostic experiments

Diagnostics test specific experimental questions without being promoted to accepted benchmark
evidence. They may use bounded datasets or otherwise prioritize diagnosis over benchmark scale.

## Recorded diagnostics

| Version | Run | Status | Main result |
|---|---|---|---|
| v0.1 | [`2026-09-21-wsl2-rtx4060ti-seeds0-4`](v0.1/2026-09-21-wsl2-rtx4060ti-seeds0-4/) | Complete | Transfer's mean accuracy gain over the frozen random head fell from +0.1000 at epoch 2 to +0.0063 at epoch 5. |
| v0.1 | [`2026-09-21-wsl2-rtx4060ti-label-free-alignment-seed0`](v0.1/2026-09-21-wsl2-rtx4060ti-label-free-alignment-seed0/) | Complete | Label-free alignment improved ID accuracy by +0.0938 over unaligned but trailed the random-head control by 0.2969. |
| v0.1 | [`2026-09-21-wsl2-rtx4060ti-label-free-methods-seed0`](v0.1/2026-09-21-wsl2-rtx4060ti-label-free-methods-seed0/) | Complete | Whitening, ridge, and Procrustes did not exceed the existing 0.4219 ID cosine-plus-MSE result. |
| v0.1 | [`2026-09-22-wsl2-rtx4060ti-decision-distillation-seeds0-2`](v0.1/2026-09-22-wsl2-rtx4060ti-decision-distillation-seeds0-2/) | Complete | Correct teacher distillation reached 0.6615 ± 0.0549 ID accuracy and beat the permuted-teacher control by +0.3385 ± 0.0705. |

Do not aggregate diagnostic values into project-level benchmark claims.
