# Changelog

## Unreleased

- Establish the v0.1 package skeleton and frozen-backbone Choice inference path.
- Add a matched frozen random-head control to the core transfer experiment.
- Add TinyLlama support, structured task metadata, batched frozen-state training, and stratified
  reporting for the first broad Choice/Boolean/ordered-Score cross-backbone validation.
- Add the `FrozenDecisionCore` contract with typed Choice/Noul/Score semantics, the pinned Open-Jev
  2B teacher and its frozen decision core, label-free core distillation with gradient audits, a
  random-core control, a thin pinned JevBench public-subset adapter, and the Open-Jev transfer
  experiment runner. The implementation first passed a bounded smoke test, then the five-seed,
  three-target accepted run and its bit-exact same-environment reproduction.
- Backbone extraction now requests a single logit position; hidden states are unchanged and Gemma's
  vocabulary logits no longer dominate memory on long prompts.
- Add the pre-registered final ship gate (decision 0007).
  - New components: a nonlinear `MLPDecisionCore` with a layer-scale-matched random control, a
    rank-128 `LowRankAdapter`, a target-specific distilled baseline, and a parameter-free
    `ScalarIdentityCore`.
  - Source-only core data preparation, and Open-Jev head-input capture.
  - Scripts: audit, reproduction comparison, JevBench runner, mechanical verdict, and report
    rendering.
  - Final verdict: NO-SHIP. DecPort v0.1 is not released.
- Add publication-facing repository documentation for the technical research article, citation
  metadata, an evidence map, and prepared paper-release notes.
