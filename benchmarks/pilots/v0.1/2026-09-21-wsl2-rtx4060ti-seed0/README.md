# v0.1 CUDA pilot: Windows + WSL2 + RTX 4060 Ti

## Status and conclusion

Completed successfully on 2026-09-21. The source-trained frozen head reached 0.53125
in-distribution accuracy, versus 0.37500 for the matched frozen random-head control: a gain of
0.15625, or 10 additional correct predictions among 64 examples. This is enough directional signal
to justify a larger multi-seed experiment, but the sample size, single seed, and single epoch are
not sufficient to validate the transfer hypothesis.

No full-scale benchmark was started and no result in this directory is an accepted benchmark.

## Experiment

- Repository base commit: `1a94418` (`Add frozen random-head control to experiment`).
- Source: `Qwen/Qwen3-0.6B`.
- Target: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Shared size: 256; maximum sequence length: 512.
- One epoch, learning rate 0.001, weight decay 0.01, seed 0.
- Three option-permutation trials per evaluation example.
- Frozen backbones in every condition.
- Frozen source head during transfer and frozen fresh random head in the control.
- Native, transfer, and random-control target adapters started from the same initialized state.

The exact command was:

```bash
uv run python scripts/run_experiment.py \
  --config configs/v0.1.json \
  --train data/pilot-v0.1/train.jsonl \
  --eval data/pilot-v0.1/eval.jsonl \
  --ood data/pilot-v0.1/ood_boolq.jsonl \
  --output runs/pilot-v0.1 \
  --device cuda \
  --epochs 1
```

The prepared data contained 64 SST-2 and 64 AG News training examples, 32 SST-2 and 32 AG News
ID evaluation examples, and 32 BoolQ OOD examples. See [`data_manifest.json`](data_manifest.json)
for dataset IDs, splits, and hashes.

Before dataset preparation and training, the locked project environment was synchronized with the
development and training extras. `ruff check .` passed, and pytest completed with 35 tests passed
and the two explicitly opt-in real-backbone tests skipped. The four-condition pilot itself then
exercised both real backbones and CUDA end to end.

## Results

| Condition | ID accuracy | BoolQ OOD accuracy | Epoch loss | ID option-order robustness |
|---|---:|---:|---:|---:|
| Qwen source | 0.703125 | 0.593750 | 1.152646 | 1.0 |
| SmolLM native target | 0.468750 | 0.500000 | 1.121481 | 1.0 |
| DecPort frozen source-head transfer | **0.531250** | 0.531250 | 1.116784 | 1.0 |
| Frozen random-head control | 0.375000 | 0.531250 | 1.038964 | 1.0 |

- Decision Portability Ratio: **1.133333** (`transfer / native target`).
- DecPort ID accuracy gain over random head: **+0.156250**.
- The transfer advantage did not appear on the small BoolQ OOD slice; both target frozen-head
  conditions scored 0.53125.
- The random control's lower training loss did not translate to better ID accuracy, but a one-epoch
  aggregate loss is not convergence evidence.

The complete metrics, including macro-F1, NLL, Brier score, and ECE, are in
[`results.json`](results.json).

## CUDA and WSL2 observations

PyTorch detected the RTX 4060 Ti and completed real CUDA execution. One-second board telemetry is
stored in [`gpu_usage.csv`](gpu_usage.csv).

- Total GPU memory: 8,188 MiB.
- Peak board-wide memory: 3,099 MiB.
- Pre-run Windows desktop baseline: approximately 1,019 MiB.
- Approximate peak increment during the pilot: 2,080 MiB.
- Peak GPU utilization: 67%; peak temperature: 41 C; peak power: 69.18 W.
- No CUDA error or out-of-memory event occurred.

WSL could not hardlink packages from the Linux uv cache into the repository on the mounted Windows
filesystem, so dependency installation fell back to copying. Hugging Face emitted an unauthenticated
rate-limit warning, and Triton emitted a harmless `_POSIX_C_SOURCE` redefinition warning while
building its driver extension. None affected the run.

See [`environment.json`](environment.json) for the recorded software and driver versions.

## Artifacts

The condition directories are copied verbatim from the ignored working run:

- `source/`: trained Qwen adapter and shared source head.
- `native_target/`: trained SmolLM adapter and native head.
- `transfer_target/`: trained SmolLM adapter; its frozen head is `source/head.safetensors`.
- `random_head_control/`: trained SmolLM adapter and frozen random head.

The entire retained run is 5,116,483 bytes (4.879 MiB). Base-model weights are intentionally not
included. Integrity hashes for every copied run artifact are in [`SHA256SUMS`](SHA256SUMS).
