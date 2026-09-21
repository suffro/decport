# v0.1 diagnostics

The v0.1 diagnostic series examines convergence and minimal variants of the frozen-backbone
portability experiment without promoting bounded results to benchmark evidence.

## Runs

| Date | Platform | Seeds | Epochs | Train / ID / OOD | Epoch-5 DPR | Epoch-5 transfer gain over random |
|---|---|---:|---:|---:|---:|---:|
| 2026-09-21 | [Windows + WSL2, RTX 4060 Ti](2026-09-21-wsl2-rtx4060ti-seeds0-4/) | 0–4 | 5 | 128 / 64 / 32 | 1.0328 ± 0.1144 | +0.0063 ± 0.0450 |

These are diagnostic results on the bounded pilot data, not accepted benchmark evidence.

The separate seed-0
[`label-free latent-alignment diagnostic`](2026-09-21-wsl2-rtx4060ti-label-free-alignment-seed0/)
trained only the SmolLM adapter against frozen Qwen latents. Its ID accuracy was 0.4219 versus
0.3281 unaligned and 0.7188 for the supervised frozen random-head control; label-free DPR was
0.5745.
