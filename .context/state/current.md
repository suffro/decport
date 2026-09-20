# Current State

## Current focus

The smallest credible v0.1 implementation and reproducible cross-backbone experiment path.

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

## Next

- Run a larger, multi-seed configured experiment on suitable hardware to test whether the pilot's
  transfer-over-random advantage is stable.
- Review learning curves and native-target performance before interpreting portability.
- Commit benchmark results only after a reproducibility run.

## Blockers

- No technical implementation blocker.
- The core transfer hypothesis remains experimentally unverified at meaningful scale.
