# 0001 — Shared latent decision head

## Status

Accepted for v0.1.

## Decision

Each frozen backbone exposes one last-token representation per candidate option. A small
backbone-specific `LayerNorm → Linear → GELU → Linear` adapter maps its native hidden width into a
common latent width. One `LayerNorm → Linear(1)` head scores candidates in that shared space.

The source run trains its adapter and the shared head. The native target baseline trains a separate
adapter and head. The transfer run loads and freezes the source head, then trains only a new target
adapter. A matched control freezes a fresh random head and trains another target adapter from the
same initial weights.

## Rationale

This is the smallest architecture that directly tests DecPort's portability claim while keeping the
model families frozen and isolating backbone-specific representation differences in one component.

## Consequences

- Candidate options can remain dynamic at runtime.
- The experiment can measure adapter-only transfer against a like-for-like native target baseline.
- The random-head control tests whether supervised target adaptation succeeds around any fixed head.
- Transfer is not established by architecture alone; the reported ratio must come from a meaningful,
  reproducible experiment.
