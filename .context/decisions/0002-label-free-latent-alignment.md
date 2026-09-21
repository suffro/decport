# 0002 — Label-free cross-backbone latent alignment diagnostic

## Status

Accepted as a diagnostic method for v0.1; not accepted as portability evidence.

## Decision

Test label-free transfer by training a fresh SmolLM adapter to reproduce the latent vectors emitted
by the already supervised Qwen adapter for identical `(state, question, candidate)` inputs. Use a
unit-weighted sum of cosine distance and mean squared error. Freeze both backbones, the Qwen adapter,
the Qwen decision head, and the target copy of that head. Optimize only the SmolLM adapter.

Before alignment, reconstruct every input without its answer. The alignment trainer rejects any
record with a non-null answer, does not call a decision head, and does not compute decision loss.

## Rationale

This is the smallest direct test of whether the shared latent itself can act as a label-free
cross-backbone supervision signal. A matched unaligned target adapter distinguishes alignment from
chance initialization, while the existing native, supervised-transfer, and frozen random-head
conditions keep the result connected to the core experiment.

## Consequences

- The source and target backbones must be resident during alignment, but they remain frozen.
- The bounded seed-0 diagnostic improved over the unaligned adapter but not the random-head control
  on ID data, so it does not justify a portability claim.
- Any later scale-up must retain the hard no-answer validation and matched initialization controls.
