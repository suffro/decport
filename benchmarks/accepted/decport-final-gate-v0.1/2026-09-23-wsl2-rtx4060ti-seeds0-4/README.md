# DecPort final ship gate — five seeds, three targets

**Status: accepted benchmark evidence, reproduced bit-exactly. FINAL VERDICT: NO-SHIP.**

The verdict and its answers are in [`../FINAL_REPORT.md`](../FINAL_REPORT.md). The frozen protocol
is decision 0007. All tables are in [`run/report_tables.md`](run/report_tables.md).

## Protocol (frozen at commit `d0aa96f` before the run)

- **Source system.** The frozen `ZefanCai/Open-Jev-2B@0c7aa49` supplies the exact float32 2048-d
  head input.
  - A nonlinear DecisionCore reads it: `LayerNorm(2048) → Linear(2048,512) → GELU →
    Linear(512,128) → GELU → Linear(128,1)`, with 1,118,977 parameters.
  - The core was trained once with labels on `data/jev-broad-v0.1-source-core` (4,500 source-only
    decisions, disjoint from every transfer split) and calibrated on 900 more (T = 1.0842). It was
    then frozen.
  - Core SHA-256: `8d3ec3087e8cefe94f45562723b3bc851c523a245a1a3619c0f0b3104770000d`.
- **Targets.** Frozen SmolLM2-360M-Instruct, Gemma 3 270M Instruct, and TinyLlama-1.1B-Chat. The
  only trainable part is the adapter `LayerNorm(d) → Linear(d,128,no bias) → Linear(128,2048,no
  bias)`: 386,944 / 345,344 / 528,384 parameters.
- **Transfer.** Label-free. The objective is `KL(source ‖ target)` at T = 1 plus `0.1 ×
  centered-logit MSE` on calibrated typed logits, over the unchanged 4,500 transfer-train inputs
  of `data/jev-broad-v0.1`.
  - Training: 10 epochs, AdamW, lr 1e-3, batch 8, seeds 0–4.
  - Evaluation: the unchanged 2,000 ID and 1,500 OOD decisions.
- **Conditions per (seed, target).**
  - A: learned core.
  - B: random core, one per seed. Gaussian tensors are norm-matched, then layer-wise mean/std
    matched to the learned core on source representations.
  - C: mismatched teacher, deranged within kind and width. One fixed point per seed: the single
    5-option Choice.
  - D: untrained adapter.
  - E: target-specific module, the core architecture on target states (395,265 / 559,745 /
    1,118,977 parameters). It is trained on the same outputs with the same objective.
  - A–D share one adapter initialization.

## Deviations

There were none after the full run started, and no run was restarted. The two pre-run corrections
from the smoke test are recorded in decision 0007:

- 2 ARC-Easy duplicates were skipped during source-data preparation.
- Layer-wise scale matching was added to the random core. With norm matching alone, its output
  scale was about 350× too small.

## Integrity (`audit.json`, `audit_failure_path.json`)

`scripts/audit_final_gate.py --verify-core` passed on the run. It verified:

- the frozen config (SHA-256 `c93e8914…`) against the run;
- data hashes and composition for all five splits;
- source-core disjointness;
- Open-Jev head parity on every captured representation (max abs diff 0.0);
- an unchanged Open-Jev LoRA/head digest;
- the learned core: unchanged across every condition, reloaded from `run/source_core/` to the
  same digest, with recomputed source outputs identical (0.0);
- seed-specific random cores that differ from the learned core;
- target backbone digests unchanged;
- matched A–D initialization;
- gradients only in the adapter or target module (4 or 8 tensors), never in a core or backbone;
- zero answers in distillation and teacher inputs, all 4,500 transfer-train answers stripped at
  load, and the label guard fired;
- every decision counted in every per-type and per-dataset metric;
- artifact hashes;
- all numbers finite.

Six tampered copies were each rejected with the expected message: an injected core gradient, a
dropped Noul decision, a labeled distillation record, a broken initialization match, altered
adapter bytes, and a changed core digest.

## Reproduction (`reproduction/`)

- The identical command, commit, config, data, and hardware were re-executed from scratch.
- `reproduction_check.json` records a pass:
  - 93 result JSON files, maximum numeric difference 0.0;
  - 81/81 safetensors artifacts byte-identical;
  - criteria A–C and E identical.
- The reproduction's own audit passed. Its artifacts are byte-identical and are not duplicated
  here.
- Wall time was 72 min for the run and 73 min for the reproduction.
  - The source representation pass took 23 min: 44,624 prompts, 6.57M tokens, 4,399 MiB peak.
  - Target caching took 2–5 min per target.
  - Each seed took about 7.3 min.

## JevBench v1.4.0 — public subset (231/534) (`jevbench_public/`)

Only the 231 public tasks exist locally; these are not full-JevBench results.

- Each (target, condition) used the adapter selected by the pre-registered label-free rule:
  lowest final-epoch training loss, ties to the lowest seed, and seed 0 for the untrained adapter
  (`selection.json`).
- Per-system reports, provenance, and hashes of the per-item records kept out of Git are included.
- Summary: source system 137/231. Learned-core DecPort scores SmolLM2 89, Gemma 73, and
  TinyLlama 74 (37 context refusals), against 73.4 expected from uniform guessing.

## Commands

```bash
uv sync --extra dev --extra train --extra openjev
uv run python scripts/prepare_source_core_data.py --output data/jev-broad-v0.1-source-core \
  --broad-data data/jev-broad-v0.1 --seed 0
bash scripts/run_final_gate_wsl_archive.sh data/jev-broad-v0.1 data/jev-broad-v0.1-source-core \
  runs/decport-final-gate-v0.1-seeds0-4 configs/decport-final-gate-v0.1.json
uv run python scripts/audit_final_gate.py --run runs/decport-final-gate-v0.1-seeds0-4 \
  --data data/jev-broad-v0.1 --source-data data/jev-broad-v0.1-source-core \
  --config configs/decport-final-gate-v0.1.json --verify-core
uv run python scripts/compare_final_gate_runs.py --first runs/decport-final-gate-v0.1-seeds0-4 \
  --second runs/decport-final-gate-v0.1-seeds0-4-repro --output reproduction_check.json
uv run python scripts/run_final_gate_jevbench.py --jevbench-root ../jevbench \
  --run runs/decport-final-gate-v0.1-seeds0-4 --output runs/decport-final-gate-v0.1-seeds0-4-jevbench
uv run python scripts/decide_final_gate.py --run runs/decport-final-gate-v0.1-seeds0-4 \
  --jevbench runs/decport-final-gate-v0.1-seeds0-4-jevbench/summary.json --audit audit.json \
  --reproduction reproduction_check.json --output verdict.json
```

`../jevbench` must be a clean checkout of `2fa63fa3226cb369795525ed011800f57dcbd894`.

## Archive contents

- `run/` is the complete run:
  - configs, both data manifests, and provenance;
  - telemetry, GPU samples, and start/end times;
  - `source_core/`: the learned core (`core.safetensors`) and its training and calibration record;
  - `source_system/`: source metrics, teacher logits, and Open-Jev raw logits;
  - per-seed `results.json`, random cores, and all 75 adapter/module artifacts;
  - `aggregate_summary.json` and `report_tables.md`.
- `reproduction/`: the reproduction's aggregate, provenance, telemetry, GPU samples, audit, and the
  comparison.
- `jevbench_public/`: JevBench's own reports and provenance for 16 systems, the selection,
  `summary.json`, and `excluded_raw_artifacts.json`.
- `audit.json`, `audit_failure_path.json`, `verdict.json`, `environment.json`, and `SHA256SUMS`.
