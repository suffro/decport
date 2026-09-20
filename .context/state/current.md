# Current State

## Current focus

The smallest credible v0.1 implementation and reproducible cross-backbone experiment path.

## Recent relevant changes

- Added package/CI skeleton and the unified decision schema.
- Added real Qwen3-0.6B and SmolLM2-360M-Instruct frozen hidden-state extraction.
- Added lightweight adapters, the shared scalar head, and dynamic Choice inference.
- Added dataset conversion/preparation, option shuffling, three-mode training, evaluation metrics,
  safetensors artifacts, and the reproducible experiment runner.
- Seeded component initialization and matched native/transfer target-adapter starting weights make
  the baseline comparison controlled and repeatable.
- Verified a bounded real-model/data smoke run outside the repository. It is not benchmark evidence.

## Next

- Run the configured experiment at meaningful scale on suitable hardware.
- Review learning curves and native-target performance before interpreting portability.
- Commit benchmark results only after a reproducibility run.

## Blockers

- No technical implementation blocker.
- The core transfer hypothesis remains experimentally unverified at meaningful scale.
