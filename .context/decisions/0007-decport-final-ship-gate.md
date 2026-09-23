# 0007 — DecPort final ship gate

## Status

Protocol frozen before the full run. The outcome section is appended after the run, audit, and
reproduction. The protocol below must not change based on observed results. A restart is allowed
only for a genuine implementation or runtime bug, and every deviation is recorded under
"Deviations".

## Question

Does DecPort, as originally intended, provide a genuinely useful reusable learned decision core
across heterogeneous frozen LLM backbones through lightweight model-specific adapters?

The answer is exactly one of `SHIP` or `NO-SHIP`, decided mechanically by the pre-registered
criteria below (`decport.final_gate.evaluate_ship_gate`, committed with this record).

## Why a new core

Decision 0006 showed that Open-Jev behavior transfers label-free, but its shipped core is a
rank-one `Linear(2048 → 1)`. An expressive adapter absorbs any readout direction, so a random core
did as well as the real one. This gate replaces that core with a compact nonlinear core learned once
on the source side, and replaces the expressive adapter with a deliberately weak rank-128 linear
map. If the learned core carries reusable decision structure, the weak adapter should do much better
through it than through a random core of the same architecture and scale.

## Protocol

### Source representation (unchanged, frozen)

The released `ZefanCai/Open-Jev-2B` package (Hub `0c7aa498b1627be8da4acf34c863ff0ee0a92785`,
manifest SHA-256 `58319da5c2a948a4645e46d9c982be44867d78779ea1c3bfb81b64867f58ef3a`) on
`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`, loaded by upstream Open-Jev
`3308a15ccd7eea1df7a37d6ddc39b023b801ba16`. Backbone, LoRA, head, and calibration stay frozen.

The source representation of a candidate is the exact float32 2048-d tensor entering the upstream
Open-Jev head. A forward hook captures it on every teacher batch; the Open-Jev head re-applied to the
captured tensor must equal the upstream logits (max abs difference ≤ 1e-5, else abort). Candidate
prompts are Open-Jev's own; teacher max length 4096, candidate batch 8.

### Learned DecisionCore (source side, trained once, then frozen permanently)

```text
input 2048 → LayerNorm(2048) → Linear(2048, 512) → GELU → Linear(512, 128) → GELU → Linear(128, 1)
```

- 1,118,977 parameters, float32, exact-erf GELU, LayerNorm with affine parameters.
- Typed semantics are unchanged: Choice and Score score one candidate prompt per option/level; Noul
  scores one prompt and uses logits `[0, s]` ordered `(false, true)`.
- One shared calibration temperature, fitted after training (below).
- Training uses source-side labels only. Typed cross-entropy on uncalibrated typed logits
  (Noul `[0, s]` is binary cross-entropy). AdamW, lr 1e-3, weight decay 0.01, gradient clip 1.0,
  batch 8 decisions homogeneous in (kind, width), 10 epochs, no early stopping, no model selection.
  Initialization seed `sha256("0:source-core")[:4]`; batch-order seed 0.
- Calibration: the single temperature minimizing mean NLL on the source calibration split over the
  grid `exp(linspace(ln 0.05, ln 20, 4001))`.
- The core is trained **once**, shared by every seed, target, and learned-core condition, and
  verified by SHA-256 before and after every condition and after the run.

### Data

Transfer splits are byte-identical to the accepted benchmark (`data/jev-broad-v0.1`):

| Split | Decisions | Composition |
|---|---:|---|
| train (transfer, unlabeled) | 4,500 | 1,500 each ARC-Easy Choice / BoolQ Noul / Yelp Score |
| ID | 2,000 | 570 ARC-Easy / 700 BoolQ / 730 Yelp |
| OOD | 1,500 | 500 OpenBookQA / 500 QNLI / 500 Amazon |

SHA-256 (from `data/jev-broad-v0.1/data_manifest.json`):

- train `031edadfa128920f45d0b19e46df17103116a4c8ff5d36e0aa687fdda57c839e`
- ID `9c67d1498c27834ac268cbb047a7a82ef767119e125395ce757946864264e671`
- OOD `35bcbf9e9d00a6757e77c5e5b163ae5c1dbf868cbb0d2377b94757b5f78a8060`

**Source-core splits (new, source-only, labeled).** The core must not be trained on the transfer
train split. If it were, its outputs on those decisions would contain memorized labels, and the
"label-free" target training would in fact be supervised through the teacher. The core is therefore
trained on disjoint records from the **same three ID datasets**, prepared by
`scripts/prepare_source_core_data.py` (seed 0) into `data/jev-broad-v0.1-source-core`:

- Choice: ARC-Easy `test`, which is otherwise unused (the broad data uses `train` and
  `validation`).
- Noul: BoolQ `train` records 1,500–3,299 of the same seed-0 shuffle whose first 1,500 are the
  transfer train.
- Score: Yelp `train` records 1,500–3,299 of the same seed-0 shuffle.
- `source_train.jsonl`: 4,500 decisions (1,500 per type). `source_calibration.jsonl`: 900 decisions
  (300 per type).
- Exact content overlap with transfer train, ID, and OOD is asserted to be zero.
- File SHA-256 values are recorded under "Frozen hashes" below.

**Label boundary.**

- Transfer-train answers are stripped immediately after loading and composition checks. No code
  path trains on them, selects with them, or evaluates on them.
- ID/OOD answers are read only by evaluation.
- The hard guard in `train_core_distillation_batched` rejects any labeled record. It is exercised
  before training with a labeled *source* record, and the audit checks it.

### Targets (frozen)

SmolLM2-360M-Instruct, `unsloth/gemma-3-270m-it` (Gemma 3 270M Instruct), and
TinyLlama-1.1B-Chat-v1.0, in their native dtype (bf16). Each target encodes Open-Jev's candidate
prompt text as plain text (no chat template, no truncation, max 2048 tokens); the last-token hidden
state is cached exactly once.

### Target adapter (the only DecPort per-target trainable component)

```text
LayerNorm(d) → Linear(d, 128, bias=False) → Linear(128, 2048, bias=False)      (no activation)
```

| Target | d | Adapter parameters |
|---|---:|---:|
| SmolLM2-360M | 960 | 386,944 |
| Gemma 3 270M | 640 | 345,344 |
| TinyLlama-1.1B | 2048 | 528,384 |

### Label-free transfer objective (unchanged from 0006)

`KL(source ‖ target)` at distillation temperature 1 plus `0.1 × centered-logit MSE`, both on
calibrated typed logits. The teacher is the source system: Open-Jev representation → learned core →
calibration. Its detached outputs are computed once on the unlabeled transfer inputs. Training uses
AdamW, lr 1e-3, weight decay 0.01, clip 1.0, batch 8 homogeneous in (kind, width), and 10 epochs.
There is no early stopping and no selection. Seeds are 0, 1, 2, 3, 4.

### Conditions (per target, per seed)

| Condition | Frozen core | Trainable | Teacher outputs |
|---|---|---|---|
| A `learned_core_distillation` | learned core | rank-128 adapter | correct |
| B `random_core_distillation` | random core (below) | rank-128 adapter | correct |
| C `mismatched_teacher_distillation` | learned core | rank-128 adapter | deranged within (kind, width) |
| D `untrained_adapter` | learned core | none | — |
| E `target_specific_distillation` | none (typed assembly + learned-core temperature only) | target-specific module | correct |

- A–D start from one identical adapter state per (seed, target): seed `sha256("{seed}:{target}")`.
  The audit verifies that each condition's initial state hash is identical.
- **Random core.** The learned core's exact architecture and temperature, one per seed (seed
  `sha256("{seed}:random-core")`). No learned direction is copied. Construction:
  1. Every parameter tensor is an i.i.d. standard-normal draw rescaled to the Frobenius norm of the
     corresponding learned-core tensor.
  2. Layer-wise scale matching. In order, each Linear layer receives one scalar gain on its weight
     and bias plus one scalar shift of its bias. These make the mean and standard deviation of its
     pre-activations over all source-core training representations (source-side, no labels) equal
     the learned core's at the same layer.
  - Step 2 was added before the full run; see "Pre-run corrections".
  - Default initialization, and norm matching alone, were rejected. The core's input LayerNorm
    removes any scale the adapter could supply, so a core with a much smaller logit range would lose
    on scale, not on learned structure.
- **Mismatched teacher.** The 0006 derangement within kind and option count. A singleton group
  keeps its own output and is reported as a fixed point (expected: the one 5-option Choice, 1/4,500).
- **Target-specific baseline.** The core architecture applied directly to target states:
  `LayerNorm(d) → Linear(d, 512) → GELU → Linear(512, 128) → GELU → Linear(128, 1)`.
  - It uses the same typed assembly and the same calibration constant, and is trained from its own
    seeded initialization (`sha256("{seed}:{target}:target-specific")`).
  - It uses the same unlabeled teacher outputs, objective, optimizer, epochs, batch, and seed.
  - It does not reuse the shared core. Its parameters are 559,745 (SmolLM2), 395,265 (Gemma), and
    1,118,977 (TinyLlama). Each is at least as large as the DecPort adapter.

### Evaluation and metrics

- ID and OOD, reported separately, overall, per decision type, and per dataset.
- Metrics: accuracy, macro-F1, NLL, Brier, ECE (10 bins), Score ordinal/expected-value MAE, and Noul
  P(true) rates; teacher top-choice agreement, KL(source ‖ target), probability correlation, and
  centered-logit MSE (also on the transfer train inputs).
- Mean ± sample SD over 5 seeds, plus paired per-seed differences of A against B, C, D, and E.
- Cost: trainable parameters, safetensors artifact bytes, training seconds, and adapter + core
  inference seconds from cached states.
- Also reported:
  - the source system (Open-Jev representation + learned core) and the original Open-Jev head, on
    ID/OOD and on JevBench;
  - the learned core's calibration-split metrics and the fitted temperature.

### JevBench

JevBench v1.4.0 at `2fa63fa3226cb369795525ed011800f57dcbd894`, **public subset (231/534)** only,
through the existing thin adapter (JevBench's own builder, runner, and scoring).

- Systems:
  - the source system;
  - for each target, conditions A–E, each with one adapter/module chosen by a label-free rule
    fixed here. Trained conditions use the seed with the lowest final-epoch training loss for that
    (target, condition), with ties going to the lowest seed. The untrained adapter uses seed 0.
- Targets are capped at their context length; longer tasks are refused and scored wrong.
- Uniform-guess expectation is `Σ 1/|labels|` over the 231 tasks.

### Pre-registered SHIP criteria

Accuracy means overall accuracy. Gains are per-seed paired differences (A − control); "mean" is over
the five seeds, and "positive" means > 0.

- **A. The learned shared core matters (primary).**
  - A target passes if:
    - mean ID gain over B is ≥ +0.05;
    - mean OOD gain over B is ≥ +0.03;
    - the ID gain is positive in ≥ 4/5 seeds;
    - the OOD gain is positive in ≥ 4/5 seeds.
  - A passes if ≥ 2/3 targets pass **and** no target has a clearly negative mean ID gain over B
    (< −0.02).
- **B. Transfer is input-specific.**
  - A target passes if its mean ID gain over C is ≥ +0.05 and positive in ≥ 4/5 seeds.
  - B passes if ≥ 2/3 targets pass.
- **C. Useful under shift.**
  - A target passes if, on OOD, its mean gain over C is ≥ +0.03 and its mean gain over D is
    ≥ +0.03, each positive in ≥ 4/5 seeds.
  - C passes if ≥ 2/3 targets pass.
  - Noul failure alone does not fail C, but it is documented.
- **D. External behavior.**
  - A target passes if its selected learned-core adapter scores
    `n_correct − uniform_expected ≥ 0.05 × 231` on the JevBench public subset.
  - D passes if ≥ 2/3 targets pass.
- **E. Practical utility over target-specific distillation.**
  - For each target and each of ID and OOD, let Δ = mean accuracy(A) − mean accuracy(E).
  - The split passes if Δ > 0 (outperforms), or if Δ ≥ −0.02 and the adapter has ≤ 0.5× the
    baseline's trainable parameters (matches with substantially fewer).
  - A target passes if both ID and OOD pass.
  - E passes if ≥ 2/3 targets pass.
  - Parameter ratios are fixed by the architectures: 0.691 (SmolLM2), 0.874 (Gemma), and 0.472
    (TinyLlama). Only TinyLlama can pass through the match route.

`SHIP` requires A ∧ B ∧ C ∧ D ∧ E, a passing audit, and a passing reproduction. Anything else is
`NO-SHIP`. The automatic NO-SHIP conditions of the task statement follow from this rule:

- the learned core is approximately like, or loses to, the random core (A);
- only one backbone shows reuse (A);
- gains disappear OOD (A, C);
- the target-specific module is equally good at similar size (E);
- labels are required (audit);
- the evidence depends on seed selection (all seeds are reported; there is no selection).

The architecture, adapter, epochs, loss weights, data, and thresholds cannot change after the run
starts.

### Audit and reproduction

`scripts/audit_final_gate.py` must pass. It verifies:

- data hashes and composition;
- teacher/Open-Jev-head parity;
- LoRA/head digest before and after the teacher pass;
- the learned core digest, unchanged across every condition and reloadable from its artifact;
- source-system parity: recomputed teacher outputs equal the stored ones exactly;
- seed-specific random cores that differ from the learned core;
- target backbone digests unchanged;
- matched adapter initialization across A–D;
- gradients only in the allowed trainable tensors;
- zero answers in teacher and distillation inputs, with the label guard fired;
- every decision counted in every metric;
- the expected mismatched fixed point;
- artifact hashes;
- all numbers finite.

The failure path is exercised once on a tampered copy.

The full run is then re-executed from scratch with the identical config and commit. It passes if all
result JSONs agree (max abs numeric difference ≤ 1e-6, verdict inputs identical) and every
safetensors artifact is byte-identical. The benchmark is accepted only if the reproduction passes.

### Frozen hashes

Filled in before the full run:

- Config `configs/decport-final-gate-v0.1.json`:
  `c93e89144d743f1fac26faa8af1d124002570994aeded40236bd56710e9ba63c`
- `data/jev-broad-v0.1-source-core/source_train.jsonl`:
  `4a4b2205d8196e9ee3a340d46ccbfaf1e221c7d3ca180c4bfee6717fe5d0ffd9`
- `data/jev-broad-v0.1-source-core/source_calibration.jsonl`:
  `dfcf07f46122c354e5cd24e1db253bc76e1535b673e4781bca78d945e306dcf8`
- `data/jev-broad-v0.1-source-core/data_manifest.json`:
  `06fdfd1d8a035198178760753396145f6b9fb0c542aa191397abc7c26dc6ba89`

The protocol commit is recorded by the run's provenance (`decport.git_commit`).

## Pre-run corrections (from the smoke test, before any full run)

The smoke test (seed 0/1, 24/12/24/12/12 decisions per type, 2 epochs) checks implementation
correctness only. Its accuracy numbers were not used for any decision.

1. **Source-core data duplicates.** The first data preparation found 2 ARC-Easy `test` questions
   identical to transfer records. The preparation now skips exact duplicates and continues through
   the same shuffled pool. The skipped indices (285 and 1063) are in the source data manifest.
2. **Random-core scale.** The smoke core's raw candidate-score std on the transfer train inputs was
   3.55. The norm-matched random core's was 0.0100.
   - A trained core's output scale comes from alignment between its layers, which random
     directions do not have.
   - The core's input LayerNorm stops the adapter from compensating.
   - As first specified, the control would therefore have violated the protocol's own
     "match scale" requirement, and its failure would have been a failure of logit range.
   - Layer-wise mean/std matching on source representations (step 2 above) replaces norm matching
     alone.
   - The scale diagnostic is recorded per seed (`raw_candidate_score_std_on_train_inputs`) and is
     checked on a re-run smoke before the full run.

## Deviations

None yet (deviations are changes after the full run starts).

## Outcome

Pending.
