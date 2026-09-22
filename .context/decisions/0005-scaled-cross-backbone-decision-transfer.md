# 0005 — Scaled cross-backbone decision-transfer validation

## Status

Accepted as stronger multi-target diagnostic evidence; not accepted as a general portability claim.

## Decision

Validate the unchanged label-free decision-space distillation method on five seeds, exactly 10 times
the earlier train/ID/OOD data, and two target families. Train one Qwen source adapter/head per seed,
then use the exact same frozen Qwen head for SmolLM2-360M-Instruct and Gemma 3 270M Instruct. Keep
all hyperparameters fixed across targets and seeds and retain unaligned, permuted-teacher, native
supervised, supervised DecPort, and frozen random-head controls.

Batch-cache deterministic frozen-backbone hidden states once as an execution optimization. This
cache does not alter prompts, losses, per-decision update order, trainable components, or metrics.

## Rationale

The prior positive result used only 128/64/32 examples, three seeds, and one target. A larger,
five-seed, multi-target experiment is required to distinguish reproducible input-specific transfer
from a small-slice or single-backbone effect.

Gemma uses `unsloth/gemma-3-270m-it`, an ungated safetensors mirror of the Gemma 3 270M instruction
checkpoint, because the configured Hugging Face account did not have access to Google's gated
repository. The exact repository revision is retained in provenance.

## Result and consequences

- SmolLM ID accuracy was 0.7828 ± 0.0173 for correct distillation, versus 0.4078 ± 0.0169
  unaligned and 0.3713 ± 0.0130 permuted.
- Gemma ID accuracy was 0.7888 ± 0.0295 for correct distillation, versus 0.3938 ± 0.0266 unaligned
  and 0.3813 ± 0.0301 permuted.
- Correct-teacher gains over both negative controls exceeded +0.37 for both targets and held in all
  five seeds.
- ID teacher/student agreement exceeded 0.81 and probability correlation exceeded 0.85 for both
  distilled targets, while permuted controls stayed near unaligned behavior.
- These results are stronger evidence that input-specific Qwen decision behavior can transfer
  without target labels through the same frozen decision head across two target families.
- BoolQ OOD gains were small and variable, and NLL/Brier/ECE worsened after distillation. Do not
  claim broad task portability, OOD transfer, or calibration quality from this experiment.
