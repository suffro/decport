# v0.1 label-free cross-backbone latent-alignment diagnostic

## Status and conclusion

Completed successfully on 2026-09-21. This is a **single-seed bounded diagnostic, not an accepted
benchmark**. It uses the existing 128-example training, 64-example in-distribution evaluation, and
32-example BoolQ OOD slices.

Matching SmolLM latents to the trained Qwen latents without answer labels reduced the mean
cosine-plus-MSE alignment loss from 1.4863 to 0.7662. On the ID set, accuracy improved from 0.3281
for the matched unaligned adapter to 0.4219 for the aligned adapter (+0.0938). However, the aligned
adapter remained far below the frozen random-head control at 0.7188 (-0.2969) and the native and
supervised-transfer conditions at 0.7344. The bounded result therefore does **not** support a
portability claim. It only provides a single-seed signal that direct latent matching improves over
leaving the matched target adapter untrained.

On the small BoolQ slice, aligned accuracy was 0.5938 versus 0.5625 unaligned and 0.5312 for the
random-head control. With only 32 examples and one seed, this is exploratory and not evidence of
OOD portability.

## Label-free control

- Qwen was first trained normally for five epochs on the labeled training slice.
- Before alignment, both backbones, the trained Qwen adapter and head, and the copied Qwen head on
  the target path were frozen.
- All 128 alignment records were reconstructed with `answer=None` before entering the alignment
  trainer. The trainer rejects any record containing an answer.
- Only the SmolLM adapter was passed to AdamW. Neither decision head was called by the alignment
  loss, and no cross-entropy or other decision loss was computed.
- Runtime checks verified tensor-for-tensor that the source adapter, source head, and target copy of
  the source head were unchanged after alignment.
- Native, supervised transfer, random-head, unaligned, and label-free aligned target adapters all
  started from the same initialized adapter state.

## Configuration

- Seed: 0.
- Source: `Qwen/Qwen3-0.6B`; target: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Shared size: 256; maximum sequence length: 512.
- Decision training: five epochs, learning rate 0.001, weight decay 0.01.
- Alignment training: five epochs, learning rate 0.001, weight decay 0.01.
- Alignment loss: `1 - cosine_similarity + mean_squared_error`, with unit weights.
- All backbones stayed frozen; no LoRA or architecture change was introduced.

The exact command was:

```bash
uv run python scripts/run_alignment_experiment.py \
  --config configs/v0.1.json \
  --train data/pilot-v0.1/train.jsonl \
  --eval data/pilot-v0.1/eval.jsonl \
  --ood data/pilot-v0.1/ood_boolq.jsonl \
  --output runs/diagnostic-label-free-alignment-v0.1-seed0 \
  --device cuda \
  --decision-epochs 5 \
  --alignment-epochs 5 \
  --seed 0
```

## Alignment loss by epoch

| Epoch | Total loss | Cosine loss | MSE loss |
|---:|---:|---:|---:|
| 1 | 1.4863 | 0.0946 | 1.3917 |
| 2 | 0.9518 | 0.0589 | 0.8929 |
| 3 | 0.8348 | 0.0538 | 0.7809 |
| 4 | 0.7667 | 0.0491 | 0.7177 |
| 5 | 0.7662 | 0.0498 | 0.7165 |

## In-distribution results

| Condition | Accuracy | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Qwen source | 0.7812 | 0.7785 | 1.4051 | 0.4190 | 0.2259 |
| Native SmolLM | 0.7344 | 0.7196 | 0.9333 | 0.4413 | 0.2130 |
| Supervised DecPort transfer | 0.7344 | 0.7130 | 0.5494 | 0.3343 | 0.1264 |
| Frozen random-head control | 0.7188 | 0.7120 | 0.7025 | 0.4225 | 0.1956 |
| Unaligned SmolLM + Qwen head | 0.3281 | 0.1890 | 1.0573 | 0.6362 | 0.0926 |
| Label-free aligned SmolLM + Qwen head | 0.4219 | 0.2666 | 1.0916 | 0.6205 | 0.1632 |

The ID label-free DPR relative to native SmolLM was 0.5745. Supervised transfer DPR was 1.0000.
The label-free gain was +0.0938 over unaligned and -0.2969 versus the random-head control.

## BoolQ OOD results

| Condition | Accuracy | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Qwen source | 0.5938 | 0.5589 | 1.2749 | 0.5503 | 0.2325 |
| Native SmolLM | 0.5625 | 0.4167 | 1.6366 | 0.7193 | 0.3818 |
| Supervised DecPort transfer | 0.5625 | 0.3600 | 1.3489 | 0.6290 | 0.3210 |
| Frozen random-head control | 0.5312 | 0.3469 | 1.3959 | 0.6638 | 0.3434 |
| Unaligned SmolLM + Qwen head | 0.5625 | 0.3600 | 0.6982 | 0.5049 | 0.0527 |
| Label-free aligned SmolLM + Qwen head | 0.5938 | 0.3725 | 0.6940 | 0.4999 | 0.1463 |

The OOD label-free DPR was 1.0556, with gains of +0.0312 over unaligned and +0.0625 over the
random-head control. Those differences correspond to one and two examples, respectively.

## Artifacts and provenance

- `results.json` contains full metrics, training losses, comparison values, configuration, and the
  machine-readable label-free audit.
- Component directories contain lightweight safetensors artifacts. The unaligned, supervised
  transfer, and label-free aligned artifacts reference the unchanged `source/head.safetensors`.
- `data_manifest.json` records dataset origins, split sizes, preparation command, and hashes.
- `environment.json` records the Windows/WSL2, CUDA, package, and GPU environment.
- `gpu_usage.csv` contains 498 one-second samples. Peak utilization was 76%, peak memory was
  2814 MiB, peak temperature was 44 C, and peak power was 65.65 W.
- `SHA256SUMS` covers the retained machine-readable results, provenance, telemetry, and model
  artifacts.

The full test suite and checksum verification were run after archiving; see the repository commit
containing this diagnostic for their exact outcomes.
