# Open-Jev DecisionCore transfer v0.1

This series tests one question: can the decision behavior of the real, released Open-Jev 2B model
be transferred across heterogeneous frozen backbones by training only DecPort adapters, while one
pretrained Open-Jev decision head and calibration stay frozen and shared?

It is separate from the earlier v0.1 scalar-head series. It has its own decision record
(`.context/decisions/0006-openjev-decision-core-transfer.md`), configs
(`configs/openjev-transfer-*.json`), and runner (`scripts/run_openjev_experiment.py`).

## Runs

| Date | Platform | Kind | Seeds | Train / ID / OOD | Status |
|---|---|---|---:|---:|---|
| 2026-09-23 | [Windows + WSL2, RTX 4060 Ti](2026-09-23-wsl2-rtx4060ti-smoke-seed0/) | Smoke test + JevBench public fidelity | 0 | 72 / 36 / 36 | Pipeline verified; not evidence |

The multi-seed experiment (`configs/openjev-transfer-v0.1.json`, 4,500 / 2,000 / 1,500 decisions,
seeds 0–4) has not been run. There is no Open-Jev transfer result yet.
