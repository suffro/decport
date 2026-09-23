# Current State

## Current focus

Transfer of a real pretrained decision system: one frozen Open-Jev 2B DecisionCore reused by
SmolLM2, Gemma 3 270M, and TinyLlama through label-free trainable DecPort adapters. The five-seed
experiment is complete, reproduced bit-exactly, and accepted. It partially supports DecPort:
Choice/Score behavior transfers, the pretrained core is not shown to matter, and Noul fails. The
open question is now whether a non-trivial pretrained core is reusable.

## Recent relevant changes

- Added package/CI skeleton and the unified decision schema.
- Added real Qwen3-0.6B and SmolLM2-360M-Instruct frozen hidden-state extraction.
- Added lightweight adapters, the shared scalar head, and dynamic Choice inference.
- Added dataset conversion/preparation, option shuffling, four-mode training, evaluation metrics,
  safetensors artifacts, and the reproducible experiment runner.
- Added a frozen random-head control to distinguish source-head transfer from an adapter learning
  around any fixed head.
- Seeded component initialization and matched native/transfer/control target-adapter starting weights
  make the baseline comparison controlled and repeatable.
- Verified a bounded real-model/data smoke run outside the repository. It is not benchmark evidence.
- Ran the first bounded CUDA pilot on WSL2 with an RTX 4060 Ti using 128 training examples, 64
  in-distribution examples, 32 BoolQ examples, and one epoch. In-distribution accuracy was 0.7031
  (source), 0.4688 (native target), 0.5312 (transfer), and 0.3750 (random-head control), for a
  Decision Portability Ratio of 1.1333 and a +0.1562 transfer gain over the random head. This is
  directional pilot evidence only, not an accepted benchmark result. The report, metrics,
  telemetry, provenance, checksums, and lightweight model artifacts are retained under
  `benchmarks/pilots/v0.1/2026-09-21-wsl2-rtx4060ti-seed0/`.
- Ran a controlled five-seed, five-epoch convergence diagnostic on the same bounded data and CUDA
  setup. DecPort's mean in-distribution accuracy gain over the matched frozen random head peaked at
  +0.1000 ± 0.0525 at epoch 2 and narrowed to +0.0063 ± 0.0450 at epoch 5. This suggests the pilot's
  advantage is primarily an early optimization effect on this data, not a clearly persistent
  convergence advantage. The run remains diagnostic rather than accepted benchmark evidence and is
  retained under `benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-seeds0-4/`.
- Added a label-free latent-alignment path that strips and rejects decision labels, freezes all
  components except a new SmolLM adapter, and minimizes cosine distance plus MSE to frozen trained
  Qwen latents for identical inputs.
- Ran the bounded seed-0 alignment diagnostic for five epochs. ID accuracy improved from 0.3281
  unaligned to 0.4219 aligned, but remained below the supervised frozen random-head control at
  0.7188; label-free DPR was 0.5745. This is not portability evidence. Results and provenance are
  retained under
  `benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-label-free-alignment-seed0/`.
- Compared stronger label-free alignment methods on the same bounded seed-0 setup. The existing
  cosine-plus-MSE result reproduced at 0.4219 ID accuracy; ridge and orthogonal Procrustes each
  reached 0.4062, and source-whitened matching reached 0.2969. Ridge produced the lowest raw latent
  loss without improving frozen-head accuracy. No method materially exceeded the current result,
  so this remains negative diagnostic evidence rather than a portability claim. Results are under
  `benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-label-free-methods-seed0/`.
- Ran a three-seed, five-epoch label-free decision-space distillation diagnostic on the unchanged
  128/64/32 slices. Correct teacher distillation reached 0.6615 ± 0.0549 mean ID accuracy, versus
  0.3802 ± 0.0477 for cosine-plus-MSE latent alignment, 0.3229 ± 0.0180 for deterministic
  permuted-teacher distillation, and 0.3438 ± 0.0563 unaligned. The +0.3385 ± 0.0705 gain over the
  permuted control is evidence that input-specific Qwen decision behavior was transferred on this
  bounded ID diagnostic. BoolQ OOD gains were small and calibration worsened, so this is not a
  general portability claim. Results are under
  `benchmarks/diagnostics/v0.1/2026-09-22-wsl2-rtx4060ti-decision-distillation-seeds0-2/`.
- Ran the larger five-seed, two-target decision-transfer validation on disjoint 1,280/640/320
  train/ID/OOD sets, exactly 10 times the earlier slices. With one exact frozen Qwen head shared by
  both targets per seed, label-free distillation reached 0.7828 ± 0.0173 ID accuracy on SmolLM and
  0.7888 ± 0.0295 on Gemma 3 270M. SmolLM unaligned/permuted accuracy was 0.4078/0.3713; Gemma
  unaligned/permuted accuracy was 0.3938/0.3813. Correct-teacher gains exceeded +0.37 over both
  controls for both targets and held in every seed. ID teacher agreement/correlation rose to
  0.8169/0.8533 for SmolLM and 0.8113/0.8565 for Gemma, while KL fell to 0.2213 and 0.2003.
  This is stronger evidence for input-specific cross-backbone decision transfer without target
  labels. BoolQ gains remained small/noisy and calibration worsened, so it is not a general
  portability or OOD claim. Results are under
  `benchmarks/diagnostics/v0.1/2026-09-23-wsl2-rtx4060ti-decision-transfer-scale-seeds0-4/`.
- Ran the first broad Jev-like five-seed validation on 4,500 train, 2,000 ID, and 1,500 OOD
  decisions spanning Choice (ARC-Easy/OpenBookQA), Boolean (BoolQ/QNLI), and ordered Score
  (Yelp/Amazon), adding TinyLlama-1.1B as a third target. Correct Qwen behavior beat both the
  untrained and mismatched-teacher controls overall on all three targets: ID gains over mismatch
  were +0.1474 Gemma, +0.1757 SmolLM, and +0.1895 TinyLlama; OOD gains were +0.0661, +0.0851,
  and +0.0712. The result is positive across Choice and Score on all three targets ID, and Score
  remains positive OOD on all three. Boolean does not reliably beat the mismatched control, and
  Gemma Choice OOD is mixed. The audit passed; the full archive is under
  `benchmarks/diagnostics/v0.1/2026-09-23-wsl2-rtx4060ti-jev-broad-seeds0-4/`. It remains
  historical diagnostic evidence for the synthetic scalar-head setup.
- Implemented the Open-Jev DecisionCore path (decision 0006):
  - The pinned `ZefanCai/Open-Jev-2B` teacher (Hub `0c7aa49`, Qwen3.5-2B `15852e8`, Open-Jev
    code `3308a15`) is loaded by the upstream loader through the new `openjev` extra.
  - Its trained `Linear(2048→1)` head and temperature 1.5188 form one frozen `LinearDecisionCore`,
    proven equal to the upstream head on every teacher batch.
  - Typed Choice/Noul/Score keep Open-Jev semantics; Noul is `[0, s]` from one prompt.
  - Adapters map `d → 256 → 2048`.
  - Label-free core distillation has label, trainability, and gradient guards.
  - Controls: untrained adapter, mismatched teacher within type, and a random frozen core.
  - A thin pinned JevBench (`2fa63fa`, v1.4.0) public-subset adapter.
- Ran the bounded smoke test (seed 0, 72/36/36 decisions, two epochs, all three targets) and
  archived it under `benchmarks/pilots/openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-smoke-seed0/`:
  - Core parity was exactly 0.0.
  - Teacher distributions were distinct and non-degenerate.
  - Gradients reached only adapter tensors.
  - All types and controls executed.
  - It is not evidence.
- On the 231-task JevBench public subset, the DecPort-loaded teacher scored 151/231 with Brier
  0.47518. Open-Jev published 150/231 and 0.4751 for the same release. Original/easy tiers match
  and one hard item differs; five near-ties below a 0.0062 margin make bf16 runtime differences
  plausible. Two smoke-ported targets also ran all 231 tasks through the frozen core; TinyLlama's
  37 over-context tasks were refused and scored wrong, not truncated.
- Backbone extraction now passes `logits_to_keep=1`. Hidden states were verified bit-identical on
  the three targets, and Gemma 3's 262k-vocabulary logits no longer drive memory past 8 GB on long
  prompts.
- Ran the five-seed Open-Jev transfer experiment exactly as configured. Its archive is the first
  accepted benchmark evidence:
  `benchmarks/accepted/openjev-transfer-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/`.
  - One operational fix: the train split's single 5-option Choice decision cannot be deranged
    within kind and width. It keeps its own teacher output as a recorded fixed point (1/4,500).
  - `scripts/audit_openjev_results.py` passed, and its failure path was exercised. The
    reproduction was bit-exact (all 75 artifacts, every result value, the teacher logits).
  - The correct teacher beats the mismatched teacher overall on every target in 5/5 seeds: +0.18
    ID and +0.07 OOD. Correct-condition accuracy is 0.448–0.466 ID and 0.376–0.393 OOD, against
    the teacher's 0.731 / 0.647. Choice and Score carry the effect; Score is the only type robust
    OOD on all targets.
  - The random frozen core is within noise of the Open-Jev core (ID +0.001 to +0.029). This gives
    no evidence that the pretrained rank-one core matters.
  - Noul fails. SmolLM2/TinyLlama learn no input-specific Noul behavior even on training data and
    collapse to "false". Gemma learns a little.
  - OOD calibration is worse than the untrained adapter on every target.
  - JevBench public subset (231/534): teacher 151 (reproduced); ported SmolLM2 92, Gemma 81,
    TinyLlama 78 (37 context refusals), against about 73 expected from uniform guessing.

## Next

- Test whether a *non-trivial* pretrained core is reusable. Move the frozen core boundary below the
  rank-one head, so the core includes Open-Jev's top transformer layers, final norm, and LoRA plus
  its head. The adapter would then map into that layer's residual stream. Keep the same data,
  seeds, and controls, and use a random-weight core of the same architecture as the decisive
  control. This is a new design and needs its own decision record before implementation.
- Diagnose Noul separately. The Noul logit is absolute (`[0, s]`), not relative, and students fit
  only the marginal P(true). Check whether the teacher's Noul score is recoverable from frozen
  target states at all, for example with a label-free probe fit to teacher scores, before changing
  the objective or loss balance.
- Keep the earlier scalar-head broad validation as historical evidence; do not reinterpret it.

## Blockers

- No technical implementation blocker.
- Unverified: reuse of any pretrained core beyond the adapter (the rank-one boundary cannot show
  it), Noul transfer, OOD Choice on Gemma, and OOD calibration of ported targets.
- JevBench's private/judge tiers are unavailable, so only public-subset (231/534) results can exist.
- The teacher's JevBench reproduction is near-exact, not bit-exact, relative to Open-Jev's published
  runtime.
