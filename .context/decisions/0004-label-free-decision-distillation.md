# 0004 — Label-free decision-space distillation

## Status

Accepted as a bounded diagnostic method; not accepted as general portability evidence.

## Decision

Train a fresh SmolLM adapter to reproduce the candidate distribution produced by the frozen,
supervised Qwen adapter and decision head for the same unlabeled input. Use temperature-scaled KL
divergence plus a small MSE penalty on per-decision centered logits. Freeze both backbones, the
Qwen adapter/head, and the target copy of the Qwen head; optimize only the SmolLM adapter.

Use a deterministic teacher-permutation control that cyclically deranges cached teacher outputs
within equal candidate-count groups. All target conditions start from the same adapter state.

## Rationale

Latent reconstruction may spend capacity matching directions that the decision head does not use.
Decision-space distillation directly supervises the behavior that must survive a backbone change,
while the permuted-teacher control tests whether any gain depends on input-specific Qwen decisions
rather than a generic optimization signal.

## Consequences

- Distillation inputs must have `answer=None`; labeled inputs are rejected at runtime.
- Teacher outputs are collected under inference mode and detached before target optimization.
- The objective is `T² × KL(teacher || student) + 0.1 × centered-logit MSE`, with `T=2.0` in the
  bounded diagnostic.
- On seeds 0–2, correct distillation reached 0.6615 ± 0.0549 ID accuracy, versus 0.3802 ± 0.0477
  for latent alignment, 0.3229 ± 0.0180 for permuted-teacher distillation, and 0.3438 ± 0.0563
  unaligned.
- BoolQ OOD accuracy improved only modestly and calibration worsened substantially, so this result
  supports input-specific decision transfer on the bounded ID slice but not a broad portability or
  OOD-calibration claim.
