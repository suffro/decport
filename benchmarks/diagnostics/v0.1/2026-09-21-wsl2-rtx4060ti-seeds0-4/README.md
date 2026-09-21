# v0.1 multi-seed convergence diagnostic: Windows + WSL2 + RTX 4060 Ti

## Status and conclusion

Completed successfully on 2026-09-21. This is a **diagnostic experiment, not an accepted
benchmark**. It reuses the bounded 128-example training and 64-example in-distribution evaluation
slices from the first pilot, so it measures convergence behavior rather than establishing a
meaningful-scale transfer result.

The DecPort transfer condition showed an early mean accuracy advantage over the matched frozen
random-head control, peaking at **+0.1000 ± 0.0525** at epoch 2. That advantage narrowed to
**+0.0063 ± 0.0450** at epoch 5. Final seed-level gains were 0.0000, 0.0000, +0.0469, +0.0469, and
-0.0625. On this diagnostic, DecPort's advantage therefore did **not** persist clearly as training
converged; the evidence is more consistent with an early optimization advantage than a stable
asymptotic advantage.

## Controlled experiment

- Seeds: 0, 1, 2, 3, 4; five epochs per condition and seed.
- Source: `Qwen/Qwen3-0.6B`; target: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Shared size: 256; maximum sequence length: 512.
- Learning rate: 0.001; weight decay: 0.01.
- Frozen backbones in all four conditions.
- Frozen fully trained source head during transfer; frozen fresh random head in the control.
- Native, transfer, and random-control target adapters used the same initialized adapter state
  within each seed.
- The architecture, prompt, datasets, and option shuffling were unchanged.
- Per-epoch curves use in-distribution evaluation without permutation repeats. The normal final
  evaluation still uses three option-permutation trials and includes the BoolQ OOD slice.

The exact command was:

```bash
uv run python scripts/run_experiment.py \
  --config configs/v0.1.json \
  --train data/pilot-v0.1/train.jsonl \
  --eval data/pilot-v0.1/eval.jsonl \
  --ood data/pilot-v0.1/ood_boolq.jsonl \
  --output runs/diagnostic-convergence-v0.1-seeds0-4 \
  --device cuda \
  --epochs 5 \
  --seeds 0 1 2 3 4
```

`loss` below is evaluation negative log-likelihood. Each seed's `results.json` also records the
mean training loss for every epoch. Values are mean ± sample standard deviation across five seeds.

## Epoch-by-epoch accuracy and portability

| Epoch | Native accuracy | DecPort accuracy | Random accuracy | DPR | DecPort gain over random |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3688 ± 0.0686 | 0.3938 ± 0.1242 | 0.3719 ± 0.0356 | 1.0747 ± 0.2895 | +0.0219 ± 0.1228 |
| 2 | 0.5219 ± 0.0754 | 0.6375 ± 0.0523 | 0.5375 ± 0.0846 | 1.2382 ± 0.1846 | +0.1000 ± 0.0525 |
| 3 | 0.6313 ± 0.0559 | 0.6875 ± 0.0552 | 0.6094 ± 0.0635 | 1.0960 ± 0.1311 | +0.0781 ± 0.1025 |
| 4 | 0.6469 ± 0.0922 | 0.6781 ± 0.0548 | 0.6219 ± 0.0792 | 1.0559 ± 0.0745 | +0.0563 ± 0.0392 |
| 5 | 0.6906 ± 0.0474 | 0.7094 ± 0.0360 | 0.7031 ± 0.0247 | 1.0328 ± 0.1144 | +0.0063 ± 0.0450 |

## Epoch-by-epoch macro-F1 and loss

| Epoch | Native F1 | DecPort F1 | Random F1 | Native loss | DecPort loss | Random loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2232 ± 0.0871 | 0.2550 ± 0.1626 | 0.2389 ± 0.0539 | 1.0644 ± 0.0813 | 1.2170 ± 0.2270 | 1.0660 ± 0.0763 |
| 2 | 0.4261 ± 0.0767 | 0.5748 ± 0.0741 | 0.4525 ± 0.1127 | 1.1305 ± 0.1810 | 0.8382 ± 0.0830 | 0.9881 ± 0.1847 |
| 3 | 0.5883 ± 0.0614 | 0.6423 ± 0.1080 | 0.5491 ± 0.0750 | 0.8567 ± 0.1145 | 0.8229 ± 0.2131 | 0.8434 ± 0.1070 |
| 4 | 0.6017 ± 0.1204 | 0.6386 ± 0.0706 | 0.5768 ± 0.0930 | 0.8722 ± 0.1840 | 0.8803 ± 0.1997 | 0.8416 ± 0.1362 |
| 5 | 0.6645 ± 0.0545 | 0.6803 ± 0.0520 | 0.6652 ± 0.0340 | 0.8791 ± 0.2022 | 0.7921 ± 0.1066 | 0.7462 ± 0.0526 |

## Artifacts and provenance

- [`convergence_summary.json`](convergence_summary.json) contains the machine-readable aggregate.
- `seed-0/` through `seed-4/` contain complete per-seed metrics and lightweight safetensors
  artifacts for all four conditions.
- [`data_manifest.json`](data_manifest.json) records the reused dataset provenance and hashes.
- [`environment.json`](environment.json) records the software, WSL2, CUDA, and GPU environment.
- [`gpu_usage.csv`](gpu_usage.csv) contains one-second board telemetry for the whole run.
- [`SHA256SUMS`](SHA256SUMS) covers the retained machine-readable results, provenance, telemetry,
  and model artifacts.

Before the run, Ruff passed and pytest completed with 37 tests passed and the two opt-in real-model
tests skipped. After the run, all 25 seed-epoch rows were checked for seed/epoch identity, required
metrics, DPR and gain arithmetic, and expected frozen-head artifact layout.
