# Changelog

## Unreleased

- Establish the v0.1 package skeleton and frozen-backbone Choice inference path.
- Add a matched frozen random-head control to the core transfer experiment.
- Add TinyLlama support, structured task metadata, batched frozen-state training, and stratified
  reporting for the first broad Choice/Boolean/ordered-Score cross-backbone validation.
- Add the `FrozenDecisionCore` contract with typed Choice/Noul/Score semantics, the pinned Open-Jev
  2B teacher and its frozen decision core, label-free core distillation with gradient audits, a
  random-core control, a thin pinned JevBench public-subset adapter, and the Open-Jev transfer
  experiment runner (smoke-tested only).
- Backbone extraction now requests a single logit position; hidden states are unchanged and Gemma's
  vocabulary logits no longer dominate memory on long prompts.
