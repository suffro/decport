# Architecture

## Overview

DecPort is a Python/PyTorch proof of concept for transferring a learned decision head between frozen
causal language-model backbones. Qwen3-0.6B is the source model; SmolLM2-360M-Instruct, Gemma 3
270M Instruct, and TinyLlama-1.1B-Chat are validated target models.

A second, separate experiment path uses a real external decision system as the source: the released
Open-Jev 2B model is the teacher, and its trained decision head plus saved calibration are one frozen
`DecisionCore` reused by every target through a trainable DecPort adapter.

## Major components

- `src/decport/backbones/`: one small interface, shared Hugging Face hidden-state extraction, Qwen,
  SmolLM, and Gemma wrappers, plus an exact frozen-representation cache for scaled experiments.
  Backbones can also encode prompts rendered by an external core (`encode_prompts`), and can refuse
  over-length prompts instead of truncating them (`allow_truncation=False`).
- `src/decport/adapter.py`: the only representation-dependent trainable component. Its optional
  intermediate width (`hidden_size`) keeps it light when the frozen core reads a wide input.
- `src/decport/head.py`: the small shared scalar head.
- `src/decport/decision_core.py`: `TypedDecision` (one Jev-wire Choice/Noul/Score question) and the
  `FrozenDecisionCore` contract: per-candidate scalars, typed-logit assembly, saved calibration,
  freeze assertions, and a state digest. `LinearDecisionCore` is the scalar linear readout form.
- `src/decport/openjev.py`: the Open-Jev boundary. It pins the Open-Jev repository commit, the
  `ZefanCai/Open-Jev-2B` package revision/manifest, and the Qwen base revision; verifies package
  hashes; loads `head.pt` + `temperature.json` as the frozen core; renders candidate prompts with
  upstream `jev.api`; and wraps the upstream `DecisionModel` as a frozen teacher.
- `src/decport/jevbench.py`: thin adapter over a clean checkout of pinned JevBench; JevBench's own
  task loader, request builder, runner, scoring, and summaries are imported, never copied.
- `src/decport/openjev_experiment.py` and `scripts/run_openjev_experiment.py`: the Open-Jev
  DecisionCore transfer protocol; `scripts/audit_openjev_results.py` audits a full run's invariants
  and completeness; `scripts/render_openjev_report.py` renders its tables;
  `scripts/run_jevbench_public.py` runs the public JevBench subset for the teacher or a ported
  target.
- `src/decport/final_gate.py` (decision 0007): the final ship gate.
  - A nonlinear `MLPDecisionCore` (`decision_core.py`) is trained once with labels on source-only
    Open-Jev representations and then frozen.
  - Targets reach it through a weak `LowRankAdapter` (`adapter.py`, rank-128 linear).
  - Controls:
    - a layer-scale-matched random core;
    - a mismatched teacher;
    - an untrained adapter;
    - a target-specific `TargetDecisionModule` scored through a parameter-free
      `ScalarIdentityCore`.
  - `evaluate_ship_gate` holds the pre-registered SHIP criteria.
  - Scripts:
    - `prepare_source_core_data.py` builds the source-only labeled splits;
    - `run_final_gate.py` and `run_final_gate_wsl_archive.sh` run the gate;
    - `audit_final_gate.py` audits a run;
    - `compare_final_gate_runs.py` checks the reproduction;
    - `run_final_gate_jevbench.py` runs the public JevBench subset;
    - `decide_final_gate.py` applies the criteria;
    - `render_final_gate_report.py` renders the tables.
  - `OpenJevTeacher.raw_logits_and_representations` returns the parity-checked 2048-d head inputs.
- `src/decport/model.py`: dynamic Choice scoring and Boolean/basic Score wrappers.
- `src/decport/data.py`: canonical records, JSONL IO, dataset converters, and option shuffling.
- `src/decport/train.py`: source/native or adapter-only training.
- `src/decport/eval.py` and `metrics.py`: ID/OOD and calibration/robustness evaluation.
- `src/decport/serialization.py`: safetensors artifacts.
- `src/decport/experiment.py`: source, native-target, transfer, and random-head control sequence.
- `src/decport/alignment.py`: label-free iterative, whitened, ridge, and Procrustes matching with a
  hard no-answer guard.
- `src/decport/alignment_experiment.py`: bounded source/native/transfer/control/alignment diagnostic.
- `src/decport/distillation.py`: label-free teacher-output collection, decision-space distillation,
  deterministic permuted-teacher control, and teacher/student behavior metrics.
- `src/decport/broad.py`: cached-state minibatch training plus dataset/task/type-stratified metrics
  for Choice, Boolean, and ordered Score decisions.
- `scripts/run_distillation_experiment.py`: controlled multi-seed distillation diagnostic and
  aggregate summary.
- `src/decport/scale_experiment.py` and `scripts/run_scale_experiment.py`: shared-source,
  two-target, five-seed scaled validation with per-target controls and aggregation.
- `src/decport/broad_experiment.py` and `scripts/run_broad_experiment.py`: fixed 4,500/2,000/1,500
  three-target, three-decision-type, five-seed broad validation and recursive aggregation.

## Data flow

Each candidate option is rendered with the same state and question and encoded independently. The
last non-padding hidden state is mapped by a backbone-specific adapter into a 256-dimensional shared
space by default. The shared head produces one scalar per candidate, and softmax is applied across
the runtime option set.

### Open-Jev DecisionCore path

```text
Open-Jev 2B teacher (all frozen):
  Open-Jev prompt ─ Qwen3.5-2B chat template ─ Qwen3.5-2B + LoRA ─ last token (2048) ─┐
                                                                                     │ captured,
Target (only the adapter trains):                                                    │ parity-checked
  Open-Jev prompt text ─ frozen target LM ─ last token (d) ─ adapter d→256→2048 ─┐   │
                                                                                 ▼   ▼
                     ONE frozen Open-Jev DecisionCore: Linear(2048→1) + typed assembly + T=1.5188
```

- The core boundary is exactly the upstream module boundary: the trained float32 `Linear(2048, 1)`
  head, its saved calibration temperature, and typed assembly. Qwen's final norm and the LoRA stay
  inside the frozen teacher backbone.
- Choice and Score score one Open-Jev candidate prompt per option/level. Noul scores one prompt and
  uses logits `[0, s]` ordered `(false, true)`; it is never mapped to a two-option Choice.
- Teacher logits come from upstream `DecisionModel.forward`; a forward hook captures the exact head
  input and the DecPort core re-scores it (max difference above 1e-5 aborts).
- Targets encode Open-Jev's candidate prompt text as plain text, without their own chat templates
  and without truncation. The teacher uses its native Qwen chat template.
- Transfer minimizes `KL(teacher_calibrated || student_calibrated)` at distillation temperature 1
  plus `0.1 × centered-logit MSE` on calibrated logits.

## External systems

- Hugging Face Transformers supplies Qwen3-0.6B, SmolLM2-360M-Instruct, Gemma 3 270M Instruct, and
  TinyLlama-1.1B-Chat.
- Hugging Face Datasets supplies SST-2/AG News/BoolQ for earlier experiments and ARC-Easy, BoolQ,
  Yelp Review Full, OpenBookQA, QNLI, and Amazon Reviews Multi for the broad validation.
- Artifacts use safetensors; no pickle checkpoint format is used. The only pickle-format file read is
  Open-Jev's own `head.pt`, loaded with `torch.load(weights_only=True)` after its package hash is
  verified.
- Open-Jev (github.com/Zefan-Cai/Open-Jev at `3308a15…`) is an optional dependency (`openjev`
  extra, with PEFT 0.19.1 and Accelerate 1.13.0). The released `ZefanCai/Open-Jev-2B` package
  (`0c7aa49…`) and `Qwen/Qwen3.5-2B` (`15852e8…`) are downloaded from the Hub at pinned revisions.
- JevBench (github.com/fstandhartinger/jevbench, `2fa63fa…`, v1.4.0) is an external clean checkout
  passed by path. Only its 231-task public subset is available.

## Important constraints

- Both LLM backbones stay frozen and in evaluation mode during training.
- No LoRA is used in v0.1.
- The source/shared head is frozen during target transfer.
- The random-head control freezes a fresh head and starts from the same target-adapter weights.
- Label-free alignment freezes both backbones, the trained source adapter/head, and the target copy
  of the source head; only a fresh, matched-initialization target adapter is optimized.
- Alignment inputs must have no `answer`, and the alignment trainer never invokes a decision head.
- Closed-form ridge and Procrustes maps are folded into the target adapter's final linear layer, so
  aligned artifacts retain the standard adapter-to-frozen-head inference path.
- Label-free decision distillation caches detached Qwen candidate scores, uses no answer fields or
  labeled loss, and optimizes only a matched-initialization SmolLM adapter through the unchanged
  frozen Qwen head.
- Direct logit matching is invariant to arbitrary per-decision offsets because logits are centered
  before MSE. The KL term uses temperature-softened candidate distributions.
- All backbones use the same textual decision prompt.
- In the scaled two-target experiment, the exact same trained Qwen head artifact is used by both
  targets within each seed and verified by SHA-256.
- In the broad experiment, the exact same trained Qwen head is reused by all three targets within
  each seed; all six target conditions start from one target-specific adapter state, and the strict
  label-free entry point rejects labeled records.
- Frozen hidden-state caching is permitted as an exact execution optimization: it must not change
  prompts, per-decision optimization steps, losses, trainable components, or evaluation behavior.
- Backbone extraction passes `logits_to_keep=1` because only hidden states are used. Hidden states
  were verified bit-identical to full-logit forwards on SmolLM2, Gemma 3 270M, and TinyLlama.
- Candidate order is shuffled during training and explicitly tested during evaluation.
- In the Open-Jev path, exactly one frozen core object (verified by SHA-256 before and after every
  condition) is shared by all targets; the Open-Jev backbone, LoRA, head, and calibration and every
  target backbone are frozen. `train_core_distillation_batched` rejects labeled records, asserts
  that only adapter parameters are trainable, audits which components received gradients on the
  first step, and verifies the core digest afterward.
- Open-Jev transfer controls: an untrained matched adapter, a deterministic mismatched teacher
  (derangement within one decision kind and width; a singleton kind-and-width group keeps its own
  output via `keep_singletons=True` and is reported per seed as a fixed point), and a random
  frozen core (same width, weight
  norm, bias, and temperature; random direction) trained on the correct teacher.
- JevBench results must be labeled public-subset results. Per-item records and raw request/response
  files contain JevBench task content and stay in ignored run directories; archives keep aggregate
  reports and hashes.
- Smoke-test metrics are not benchmark evidence.
