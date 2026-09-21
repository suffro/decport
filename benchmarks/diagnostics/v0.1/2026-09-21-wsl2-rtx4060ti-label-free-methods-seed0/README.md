# v0.1 stronger label-free alignment methods diagnostic

## Status and answer

Completed successfully on 2026-09-21. This is a **single-seed bounded diagnostic, not an accepted
benchmark**. It reuses the existing 128-example training, 64-example ID evaluation, and 32-example
BoolQ OOD slices.

No tested method materially exceeded the current 42.2% ID result. The original cosine-plus-MSE
baseline reproduced **0.4219** accuracy. Ridge and orthogonal Procrustes each reached **0.4062**,
while source-whitened cosine plus MSE reached **0.2969**. None recovered enough of the frozen Qwen
head's decision capability to approach native SmolLM (0.7344) or the supervised controls.

Ridge achieved the lowest raw latent loss, 0.6950 versus 0.7126 for cosine plus MSE, but lower ID
accuracy. In this diagnostic, globally reconstructing Qwen latents more closely was therefore not
sufficient to preserve the decision directions used by the frozen head.

## Controlled method comparison

- All methods started from the exact same initialized SmolLM adapter state.
- Qwen source training and the source/native/supervised-transfer/random/unaligned controls were
  unchanged. Their metrics exactly reproduced the preceding seed-0 alignment diagnostic.
- Both backbones, the trained Qwen adapter/head, and every target copy of the Qwen head were frozen.
- All 128 alignment examples were reconstructed with `answer=None`; every alignment entry point
  rejects records containing an answer.
- No alignment method calls a decision head or computes decision loss.
- Cosine+MSE and whitening optimize only the target adapter. Ridge and Procrustes solve label-free
  affine maps and fold them into only the final linear layer of the target adapter.
- Runtime checks verified tensor-for-tensor that the source adapter, source head, and each target
  head were unchanged after alignment.

## Methods

1. `cosine_mse`: the existing five-epoch unit-weighted cosine-distance plus MSE objective.
2. `whitened_cosine_mse`: the same objective after centering and whitening both sides with the
   trained Qwen latent covariance; covariance floor `1e-3`.
3. `ridge`: centered affine regression from initial SmolLM latents to Qwen latents with ridge
   coefficient `1.0`.
4. `orthogonal_procrustes`: centered rotation/reflection plus translation in the common
   256-dimensional latent space.

The exact command was:

```bash
uv run python scripts/run_alignment_experiment.py \
  --config configs/v0.1.json \
  --train data/pilot-v0.1/train.jsonl \
  --eval data/pilot-v0.1/eval.jsonl \
  --ood data/pilot-v0.1/ood_boolq.jsonl \
  --output runs/diagnostic-label-free-alignment-methods-v0.1-seed0 \
  --device cuda \
  --decision-epochs 5 \
  --alignment-epochs 5 \
  --alignment-methods cosine_mse whitened_cosine_mse ridge orthogonal_procrustes \
  --seed 0
```

## In-distribution results

`Gain` is accuracy relative to the matched unaligned target. `Raw loss` is the common unwhitened
cosine-plus-MSE score after fitting, so it is comparable across methods.

| Condition | Accuracy | Gain | Macro-F1 | NLL | Brier | ECE | Raw loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unaligned SmolLM + Qwen head | 0.3281 | — | 0.1890 | 1.0573 | 0.6362 | 0.0926 | 8.3751 |
| Cosine + MSE | **0.4219** | **+0.0938** | **0.2666** | 1.0916 | 0.6205 | 0.1632 | 0.7126 |
| Whitened cosine + MSE | 0.2969 | -0.0312 | 0.2147 | 1.0439 | 0.6273 | 0.0817 | 7.7411 |
| Ridge | 0.4062 | +0.0781 | 0.2120 | 1.5273 | 0.7982 | 0.2736 | **0.6950** |
| Orthogonal Procrustes | 0.4062 | +0.0781 | 0.2063 | **1.0369** | 0.6239 | **0.0223** | 1.2586 |

The corresponding DPR values versus native SmolLM were 0.5745 (cosine+MSE), 0.4043 (whitened),
and 0.5532 for both ridge and Procrustes. All remained at least 0.2969 accuracy below the supervised
frozen random-head control.

## BoolQ OOD results

| Condition | Accuracy | Gain | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Unaligned SmolLM + Qwen head | 0.5625 | — | 0.3600 | 0.6982 | 0.5049 | 0.0527 |
| Cosine + MSE | **0.5938** | **+0.0312** | 0.3725 | 0.6940 | 0.4999 | 0.1463 |
| Whitened cosine + MSE | 0.4375 | -0.1250 | 0.3455 | 0.6956 | 0.5025 | 0.0678 |
| Ridge | 0.5625 | 0.0000 | 0.4909 | 0.6959 | 0.5058 | 0.1389 |
| Orthogonal Procrustes | 0.5312 | -0.0312 | **0.4910** | **0.6929** | **0.4997** | **0.0286** |

The BoolQ set has only 32 examples; one accuracy step is 0.03125. These values are exploratory and
do not establish OOD portability.

## Reference conditions

| Condition | ID accuracy | ID macro-F1 | ID NLL | ID Brier | ID ECE | OOD accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen source | 0.7812 | 0.7785 | 1.4051 | 0.4190 | 0.2259 | 0.5938 |
| Native SmolLM | 0.7344 | 0.7196 | 0.9333 | 0.4413 | 0.2130 | 0.5625 |
| Supervised DecPort transfer | 0.7344 | 0.7130 | 0.5494 | 0.3343 | 0.1264 | 0.5625 |
| Frozen random-head control | 0.7188 | 0.7120 | 0.7025 | 0.4225 | 0.1956 | 0.5312 |

Full OOD macro-F1 and calibration metrics for these references are retained in `results.json`.

## Alignment objectives

The cosine-plus-MSE training loss fell from 1.4863 to 0.7662 over five epochs. The whitened
objective fell from 4.1985 to 1.8577, but its final raw loss remained 7.7411. Ridge and Procrustes
are single closed-form fits with final raw losses of 0.6950 and 1.2586.

## Artifacts and provenance

- `results.json` records all ID/OOD metrics, method losses, comparisons, configuration, and the
  machine-readable no-label/frozen-state audit.
- Nine component directories retain source, reference-control, unaligned, and aligned adapters.
  Every aligned adapter references the unchanged `source/head.safetensors`.
- `data_manifest.json` and `environment.json` record dataset and execution provenance.
- `gpu_usage.csv` contains 721 one-second samples. Peak utilization was 56%, peak memory was
  3172 MiB, peak temperature was 45 C, and peak power was 65.36 W.
- `SHA256SUMS` covers all retained machine-readable results, provenance, telemetry, and artifacts.
