# DecPort final ship gate — report

```text
FINAL VERDICT: NO-SHIP
```

The question was whether DecPort, as originally intended, provides a genuinely useful reusable
learned decision core across heterogeneous frozen LLM backbones through lightweight model-specific
adapters. It does not, on the tested models and tasks.

The protocol was frozen before the run in decision 0007
(`.context/decisions/0007-decport-final-ship-gate.md`, commit `d0aa96f`). The verdict was computed
mechanically by `evaluate_ship_gate` (`verdict.json`). The run passed its audit, and an independent
re-execution was bit-exact. Full tables are in
[`2026-09-23-wsl2-rtx4060ti-seeds0-4/run/report_tables.md`](2026-09-23-wsl2-rtx4060ti-seeds0-4/run/report_tables.md).

## What was tested

- **Source system.** Frozen Open-Jev 2B representations (the exact 2048-d head input) feed a
  nonlinear DecisionCore: `LayerNorm → 2048→512 → GELU → 512→128 → GELU → 128→1`.
  - The core has 1,118,977 parameters.
  - It was trained once, with labels, on 4,500 source-only decisions, calibrated on 900 more
    (T = 1.084), and then frozen.
  - Accuracy: ID 0.746, OOD 0.619. Open-Jev's own head scores 0.731 and 0.647.
- **Targets.** Frozen SmolLM2-360M, Gemma 3 270M, and TinyLlama-1.1B. Only a rank-128 linear adapter
  trains: `LayerNorm(d) → d→128 → 128→2048`, no bias, no activation. It learns without labels from
  the source system's calibrated outputs on the 4,500 transfer inputs.
- **Evaluation.** 2,000 ID and 1,500 OOD decisions of Choice, Noul, and Score; seeds 0–4.
- **Conditions.**
  - A: learned core.
  - B: random core of the same architecture, scale-matched layer by layer.
  - C: mismatched teacher.
  - D: untrained adapter.
  - E: a target-specific distilled module (the core's architecture on target states, no shared
    core).

## Pre-registered criteria

| Criterion | Requirement | Passing targets | Result |
|---|---|---|---|
| **A. The learned core matters** (primary) | A − B ≥ +5 pp ID and ≥ +3 pp OOD, positive in ≥ 4/5 seeds each, on ≥ 2 targets | none | **FAIL** |
| B. Input-specific | A − C ≥ +5 pp ID, positive in ≥ 4/5 seeds, on ≥ 2 targets | all 3 | pass |
| C. Useful under shift | OOD: A − C and A − D ≥ +3 pp, positive in ≥ 4/5 seeds, on ≥ 2 targets | all 3 | pass |
| D. JevBench public subset | ≥ uniform + 5 pp on ≥ 2 targets | SmolLM2 only | **FAIL** |
| E. Practical utility | Beat E, or come within 2 pp of it with ≤ 0.5× its parameters, on ≥ 2 targets | TinyLlama only | **FAIL** |

The audit passed and the reproduction passed. Three of the five criteria fail, including the primary
one.

## 1. Does learned decision-core reuse exist?

No. The learned core is indistinguishable from a random core of the same architecture and scale.
Accuracy differences, A − B (mean ± SD over 5 seeds; seeds positive):

| Target | ID | OOD |
|---|---:|---:|
| SmolLM2-360M | 0.000 ± 0.029 (1/5) | +0.006 ± 0.007 (5/5) |
| Gemma 3 270M | +0.015 ± 0.039 (2/5) | +0.000 ± 0.010 (2/5) |
| TinyLlama-1.1B | −0.015 ± 0.022 (2/5) | −0.011 ± 0.007 (0/5) |

No target comes near the +5 pp / +3 pp bar. On TinyLlama the random core is better OOD in every seed.

The learned core does give a steadier fit to the source's distributions in-distribution. ID KL to the
source is 0.49–0.54 against 0.59–0.92 for the random core, ID ECE is 0.05–0.07 against 0.10–0.12, and
seed-to-seed spread is much smaller. None of this becomes accuracy.

## 2. Does it work across heterogeneous backbones?

Label-free *behavior transfer* works on all three backbones; *core reuse* does not.

- Every target learns input-specific source behavior. ID gains over the mismatched teacher are
  +0.105 (Gemma), +0.144 (SmolLM2), and +0.169 (TinyLlama), in 5/5 seeds on every target.
- The random core and the target-specific module achieve the same, so the shared learned core is
  not what carries it.

## 3. Does it survive OOD?

Only weakly, and not through the core.

- OOD gains over the mismatched teacher shrink to +0.041 / +0.042 / +0.073. Gains over the untrained
  adapter are +0.041 / +0.043 / +0.067.
- Ported OOD accuracy is 0.351–0.378, against the source system's 0.619.
- OOD calibration is poor. ECE is 0.15–0.31; TinyLlama's NLL is 3.54 ± 1.35, against 1.39 for its
  untrained adapter.

## 4. Does it provide value over target-specific distillation?

No. The target-specific module E uses the same unlabeled teacher outputs, training time, and
objective, and matches or beats DecPort everywhere except SmolLM2 OOD.

| Target | A − E, ID | A − E, OOD | Params per target: adapter / E |
|---|---:|---:|---:|
| SmolLM2-360M | −0.006 | +0.004 | 386,944 / 559,745 (0.69×) |
| Gemma 3 270M | −0.006 | −0.016 | 345,344 / 395,265 (0.87×) |
| TinyLlama-1.1B | −0.018 | −0.004 | 528,384 / 1,118,977 (0.47×) |

- Only on TinyLlama is DecPort within 2 pp of E with at most half the per-target parameters.
- Elsewhere a target-specific module of similar size is at least as good.
- The shared core adds 1.12M parameters of its own and needs a source-side training stage.

## 5. What works by decision type

Accuracy of the learned-core DecPort condition, against the mismatched teacher and the source system:

| Type | Split | SmolLM2 | Gemma | TinyLlama | Mismatched | Source |
|---|---|---:|---:|---:|---:|---:|
| Choice | ID | 0.656 | 0.457 | 0.526 | 0.21–0.28 | 0.893 |
| Choice | OOD | 0.332 | 0.265 | 0.293 | 0.24–0.27 | 0.632 |
| Score | ID | 0.285 | 0.273 | 0.430 | 0.16–0.18 | 0.526 |
| Score | OOD | 0.263 | 0.296 | 0.365 | 0.18–0.20 | 0.534 |
| Noul | ID | 0.637 | 0.629 | 0.601 | 0.62–0.64 | 0.854 |
| Noul | OOD | 0.474 | 0.491 | 0.477 | 0.48–0.49 | 0.692 |

- Choice transfers in-distribution on every target and weakly OOD, except Gemma, which is at the
  mismatched level.
- Score transfers ID and OOD on every target, but far below the source.
- The random core and E are within noise of these numbers for every type.

## 6. What fails

- **Core reuse.** This is the hypothesis itself (criterion A).
- **Noul.** Learned-core students predict "true" for 62–100% of Noul decisions. Noul accuracy is at
  or below the near-constant mismatched control on every target ID. OOD it is within 0.02 of that
  control. The source system is 0.854 ID and 0.692 OOD.
- **Gemma Choice OOD.** It is at the mismatched level.
- **OOD calibration**, on every target.
- **External behavior on JevBench.** See section 7.

## 7. What do JevBench results show?

This is the JevBench v1.4.0 public subset (231/534), not full JevBench. Adapters were selected
without labels (lowest final training loss) before evaluation. Uniform guessing expects 73.4 correct.

| System | SmolLM2 | Gemma | TinyLlama* |
|---|---:|---:|---:|
| A learned core | **89** | 73 | 74 |
| B random core | 87 | 73 | 74 |
| C mismatched | 85 | 78 | 64 |
| D untrained | 69 | 73 | 83 |
| E target-specific | 96 | 70 | 65 |

The source system (Open-Jev representation + learned core) scores 137/231. Open-Jev's own head
scored 151 in decision 0006.

\*TinyLlama refused 37 tasks beyond its 2,048-token context; they are scored wrong.

- Only SmolLM2's DecPort adapter clears uniform + 5 pp (+6.8 pp). Gemma is −0.2 pp and TinyLlama
  +0.3 pp.
- On every target the learned core is within a few tasks of the random core. The controls reach the
  same range; the best system is target-specific SmolLM2 at 96.
- Ported Noul (32–41/74) and Score (2–4/18) are near chance.

## 8. Parameter, storage, and runtime cost

- **Per-target adapter.** 345,344–528,384 parameters, 1.38–2.11 MB float32 safetensors.
- **Shared core.** 1,118,977 parameters (4.48 MB), stored once.
- **Target-specific module.** 395,265–1,118,977 parameters (1.58–4.48 MB).
- **Training, from cached frozen states on an RTX 4060 Ti.** 27–36 s per adapter or module, the same
  for DecPort and E.
- **Inference, from cached states.** 0.13–0.31 s per 3,500 decisions for any condition. The frozen
  backbone forward pass dominates real inference.
- **One-time costs.**
  - Source representation pass: 23 min (44,624 prompts, 6.57M tokens).
  - Target caching: 2–5 min per target.
- **Per run.** Each seed took about 7.3 min. The full run took 72 min and its reproduction 73 min.

## 9. Exact evidence supporting the verdict

- `verdict.json`: the per-target criterion values, thresholds, and pass/fail flags, computed by the
  committed pre-registered code.
- `run/aggregate_summary.json` and `run/seed-*/results.json` contain every metric.
  `paired_accuracy_differences` gives the per-seed A − control values used above.
- `audit.json` records a passed audit. It covers:
  - Open-Jev head parity of 0.0;
  - a learned-core digest that is unchanged and reloadable;
  - source-output recompute parity of 0.0;
  - target backbone digests unchanged;
  - zero labels in distillation or teacher inputs;
  - gradients only in allowed tensors;
  - matched initialization;
  - every decision counted;
  - artifact hashes.

  Six tampered copies were rejected (`audit_failure_path.json`).
- `reproduction/reproduction_check.json`: 93 result files with a maximum numeric difference of 0.0,
  81/81 safetensors artifacts byte-identical, and identical criteria. The reproduction's own audit
  passed.
- `jevbench_public/`: JevBench's own reports, the label-free selection, and hashes of the per-item
  records that are kept out of Git.

## What remains true, separately from the hypothesis

This is not the DecPort hypothesis. Label-free *distillation of decision behavior* from a real
source system into heterogeneous frozen backbones works in-distribution. It is input-specific (5/5
seeds on every target) and keeps a small OOD effect. That capability does not need a shared learned
core: a random core or a small target-specific module does as well. The claim is limited to these
three backbones, these datasets, and this 4,500-decision budget.
