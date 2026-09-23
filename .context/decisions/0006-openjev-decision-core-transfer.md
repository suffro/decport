# 0006 — Open-Jev 2B DecisionCore transfer

## Status

Accepted as the implementation path for testing transfer of a real pretrained decision system.
Implemented and smoke-tested only. No transfer or portability evidence exists yet.

## Question

Can the decision behavior of a real Open-Jev model be transferred across heterogeneous frozen LLM
backbones with lightweight DecPort adapters while the same pretrained Open-Jev decision head/core
stays frozen?

## Decision

- **Teacher.** The released `ZefanCai/Open-Jev-2B` package at Hub revision
  `0c7aa498b1627be8da4acf34c863ff0ee0a92785` (package manifest SHA-256 `58319da5…`), on
  `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, loaded by the upstream
  `jev.model.DecisionModel` from Open-Jev commit `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`
  (`openjev` extra: PEFT 0.19.1, Accelerate 1.13.0, matching Open-Jev's release runtime). The
  complete teacher (backbone, rank-8 LoRA, head, calibration) is frozen and never retrained.
- **DecisionCore boundary.** The core is exactly the upstream module boundary: the trained float32
  `Linear(2048, 1)` head from `head.pt`, the saved calibration temperature (1.518796342858676), and
  Jev typed assembly. Qwen's final norm and the LoRA remain inside the frozen teacher backbone.
  `FrozenDecisionCore` is generic; `LinearDecisionCore` implements it; only the loader is
  Open-Jev-specific.
- **Proof that the core is the teacher's head.** Head weights must equal core weights exactly, and
  every teacher batch captures the upstream head input with a forward hook and re-scores it with the
  DecPort core; any difference above 1e-5 aborts.
- **Typed semantics.** Choice and Score score one Open-Jev candidate prompt per option/level. Noul
  scores one prompt and uses logits `[0, s]` ordered `(false, true)`, so `P(true) = σ(s / T)`.
  Noul is never represented as a two-option Choice. Metrics are reported per type.
- **Targets.** SmolLM2-360M-Instruct, Gemma 3 270M Instruct (`unsloth/gemma-3-270m-it`), and
  TinyLlama-1.1B-Chat, unchanged. Each frozen target encodes Open-Jev's candidate prompt text
  (plain text, no truncation), and a `BackboneAdapter(d → 256 → 2048)` maps it into the core's
  2048-wide input. One core object is shared by all targets and hashed before and after every
  condition.
- **Objective.** `KL(teacher_calibrated || student_calibrated)` at distillation temperature 1 plus
  `0.1 × centered-logit MSE`, both on calibrated typed logits. Temperature 1 makes the KL target
  literally Open-Jev's calibrated distribution; the earlier scalar-head experiments used 2.0. The
  transfer entry point rejects labeled records, asserts that only adapter parameters are trainable,
  and audits that gradients reach every adapter tensor and no frozen tensor.
- **Controls.** Untrained matched adapter; deterministic mismatched teacher (derangement within one
  decision kind and width, so types are never mixed); random frozen core with the same width, weight
  norm, bias, and temperature but a random direction, trained on the correct teacher. All
  conditions of a target start from one adapter state.
- **Data.** The existing broad Jev-like 4,500/2,000/1,500 split (ARC-Easy, BoolQ, Yelp;
  OpenBookQA, QNLI, Amazon OOD) converted to typed decisions, with BoolQ/QNLI as Noul. Labels are
  used only for evaluation.
- **JevBench.** A clean checkout of `fstandhartinger/jevbench` at `2fa63fa3226cb369795525ed011800f57dcbd894`
  (v1.4.0), verified by commit and public-file hashes. JevBench's own request builder, runner,
  scoring, and summaries are used unchanged. Only the 231-task public subset is available. Its
  scoring, task loading, and public data are byte-identical to `f8ce713`, the commit behind
  Open-Jev's published 2B result (150/231), so the teacher can be checked against that report.

## Rationale and rejected alternatives

- Reimplementing Open-Jev, retraining a head, or reusing DecPort's scalar head would change the
  scientific variable. The upstream loader and prompt renderer are imported instead.
- Mapping Noul to Choice would change Open-Jev semantics (one prompt and a fixed zero reference,
  rather than two independently scored candidates).
- Putting the final norm or LoRA into the core would move the boundary away from the trained head
  that Open-Jev actually ships and calibrates.
- Target chat templates were not introduced, to keep the validated DecPort target extraction path;
  the asymmetry with the chat-templated teacher is recorded as a limitation.
- Open-Jev's own training corpus was not used as transfer data: its licensing is mixed, its states
  are long, and the broad split keeps direct comparability with the previous broad diagnostic.
- A 256-wide adapter bottleneck keeps adapters near 0.7–1.1M parameters instead of ~4–8M.

## Known limitation

Open-Jev's core is a rank-one linear readout. The adapter's final linear layer can express any
readout direction, so the frozen core constrains the adapter mainly through its scale, bias, and
calibration. The random-core control is required to test whether the *pretrained* core matters
beyond a fixed readout. A positive teacher-versus-mismatch result alone would show transfer of
Open-Jev behavior, not that the specific core is necessary.

## Smoke outcome (2026-09-23, not evidence)

Seed 0, 72/36/36 decisions, two epochs, all three targets, RTX 4060 Ti 8 GB:

- Core parity was exactly 0.0, and teacher distributions were distinct and non-degenerate for
  every type.
- Every trained adapter received gradients on all 6 adapter tensors and on no core or backbone
  tensor; the core digest was unchanged.
- The labeled-record guard rejected its probe.
- On the public JevBench subset, the DecPort-loaded teacher scored 151/231 (Brier 0.47518) against
  Open-Jev's published 150/231 (0.4751). This is near-exact, not bit-exact.
- Smoke-ported targets produced valid JevBench distributions; over-context tasks were refused with
  422 and scored wrong.
- Gemma's full-vocabulary logits pushed caching past 8 GB on long prompts. Backbone extraction now
  passes `logits_to_keep=1`, which leaves hidden states bit-identical.

## Consequences

- Open-Jev teacher inference is the dominant cost; it runs once for all seeds and targets. Measured
  throughput is 4–5k tokens/s with the reference linear-attention kernels, about 15 minutes for the
  3.84M-token broad split. Peak allocation stays near 4.4 GB even on the longest prompts at
  candidate batch 8.
- JevBench per-item records and raw responses stay out of Git; archives keep aggregate reports and
  hashes.
- Results must not be described as portability evidence until the multi-seed experiment exists and
  has been reproduced.
