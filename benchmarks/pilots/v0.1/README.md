# v0.1 pilots

The v0.1 series tests whether the Qwen3-0.6B source-trained decision head can be frozen and reused
with a trained SmolLM2-360M adapter, compared with both a native target head and a matched frozen
random-head control.

## Runs

| Date | Platform | Seed | Epochs | Train / ID / OOD | DPR | Transfer gain over random |
|---|---|---:|---:|---:|---:|---:|
| 2026-09-21 | [Windows + WSL2, RTX 4060 Ti](2026-09-21-wsl2-rtx4060ti-seed0/) | 0 | 1 | 128 / 64 / 32 | 1.1333 | +0.15625 |

This series currently contains directional pilot evidence only. The next step is a larger,
multi-seed experiment with enough epochs to assess convergence and target-native strength.
