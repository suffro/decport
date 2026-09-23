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

The follow-up
[`stronger-method comparison`](2026-09-21-wsl2-rtx4060ti-label-free-methods-seed0/) reproduced
the 0.4219 baseline, while ridge and Procrustes reached 0.4062 and source-whitened matching reached
0.2969. No stronger method recovered more frozen-head decision capability on the bounded seed-0
run.

The three-seed
[`decision-space distillation diagnostic`](2026-09-22-wsl2-rtx4060ti-decision-distillation-seeds0-2/)
directly matched frozen Qwen candidate distributions. Correct distillation reached 0.6615 ± 0.0549
ID accuracy and exceeded the permuted-teacher control by +0.3385 ± 0.0705. BoolQ OOD accuracy gains
were small and calibration degraded, so the result is diagnostic evidence rather than a general
portability claim.

The five-seed
[`two-target scaled validation`](2026-09-23-wsl2-rtx4060ti-decision-transfer-scale-seeds0-4/)
replicated strong ID transfer on SmolLM and Gemma over ten-times-larger SST-2/AG News slices, while
BoolQ OOD remained weak.

The first
[`broad Jev-like validation`](2026-09-23-wsl2-rtx4060ti-jev-broad-seeds0-4/) uses 4,500 train,
2,000 ID, and 1,500 OOD decisions, three target families, and explicit Choice, Boolean, and ordered
Score tasks. Correct-teacher transfer beats both negative controls overall on every target and is
strong for Choice and Score, but Boolean does not reliably beat mismatched teachers. See the run
report for all dataset/type/backbone strata and calibration metrics.
