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

## Next

- Run a larger-data, independently reproduced multi-seed experiment before making portability
  claims; the bounded convergence diagnostic did not show a persistent transfer-over-random gain.
- Treat decision-space distillation as the leading bounded label-free method, but investigate its
  poor BoolQ NLL/ECE and validate on larger data before any portability claim.
- Do not prioritize further global latent matching without a new head-relevant hypothesis;
  whitening, ridge, and Procrustes did not improve the existing baseline.
- Commit benchmark results only after a reproducibility run.

## Blockers

- No technical implementation blocker.
- The core transfer hypothesis remains experimentally unverified at meaningful scale.
