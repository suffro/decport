# Pilot experiments

Pilots are bounded experiments, not benchmark claims. They exercise the real model/data path,
surface runtime problems, and provide directional evidence for deciding what to run next.

## Recorded pilots

| Version | Run | Status | Main result |
|---|---|---|---|
| v0.1 | [`2026-09-21-wsl2-rtx4060ti-seed0`](v0.1/2026-09-21-wsl2-rtx4060ti-seed0/) | Complete | Transfer beat the matched random head by 15.625 accuracy points on 64 ID examples. |
| openjev-transfer-v0.1 | [`2026-09-23-wsl2-rtx4060ti-smoke-seed0`](openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-smoke-seed0/) | Smoke test | The real Open-Jev 2B teacher and frozen core ran end to end with exact core parity, adapter-only gradients, and all three decision types. The teacher scored 151/231 on public JevBench (Open-Jev published 150/231). No transfer result. |

Do not aggregate pilot values into project-level benchmark claims. Sample counts, seeds, training
duration, and convergence evidence are intentionally limited.
