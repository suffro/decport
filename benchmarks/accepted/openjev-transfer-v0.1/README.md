# Open-Jev DecisionCore transfer v0.1 — accepted runs

This series asks whether the decision behavior of the released Open-Jev 2B model can be
transferred across heterogeneous frozen backbones by training only DecPort adapters, while one
pretrained Open-Jev decision head and calibration stay frozen and shared. The design is in decision
0006, and the smoke test is under
[`../../pilots/openjev-transfer-v0.1/`](../../pilots/openjev-transfer-v0.1/).

| Date | Platform | Seeds | Train / ID / OOD | Status |
|---|---|---:|---:|---|
| 2026-09-23 | [Windows + WSL2, RTX 4060 Ti](2026-09-23-wsl2-rtx4060ti-seeds0-4/) | 0–4 | 4,500 / 2,000 / 1,500 | Accepted; reproduced bit-exactly; partial support |
