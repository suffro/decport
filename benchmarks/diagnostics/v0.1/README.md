# v0.1 diagnostics

The v0.1 diagnostic series examines behavior of the existing four-condition portability experiment
without changing its architecture, frozen backbones, data, initialization controls, or optimization
hyperparameters.

## Runs

| Date | Platform | Seeds | Epochs | Train / ID / OOD | Epoch-5 DPR | Epoch-5 transfer gain over random |
|---|---|---:|---:|---:|---:|---:|
| 2026-09-21 | [Windows + WSL2, RTX 4060 Ti](2026-09-21-wsl2-rtx4060ti-seeds0-4/) | 0–4 | 5 | 128 / 64 / 32 | 1.0328 ± 0.1144 | +0.0063 ± 0.0450 |

These are diagnostic results on the bounded pilot data, not accepted benchmark evidence.
