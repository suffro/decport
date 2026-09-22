# v0.1 label-free decision-space distillation diagnostic

## Status and answer

Completed successfully on 2026-09-22. This is a **bounded three-seed diagnostic, not an accepted
benchmark**. It uses the unchanged 128-example training, 64-example ID evaluation, and 32-example
BoolQ OOD slices, with five epochs for source, supervised controls, latent alignment, and
decision-space distillation.

The central ID ordering held in the aggregate:

`unaligned (0.3438) < latent alignment (0.3802) < decision distillation (0.6615)`

Correct teacher distillation also materially exceeded permuted-teacher distillation (0.3229) by
**+0.3385 ± 0.0705 accuracy**. This is positive bounded evidence that input-specific Qwen decision
behavior, rather than a generic fixed-head optimization signal, was transferred to the SmolLM
adapter. It is not yet a general portability claim: BoolQ OOD accuracy improved only modestly and
its NLL, Brier, and ECE were substantially worse.

## Protocol and no-label boundary

1. Train the Qwen adapter and decision head for five supervised epochs.
2. Freeze the Qwen backbone, adapter, and head, plus the SmolLM backbone.
3. Reconstruct all 128 distillation inputs with `answer=None` and cache Qwen candidate logits under
   inference mode.
4. Initialize every relevant SmolLM adapter from the exact same saved state.
5. Train only the SmolLM adapter through the unchanged frozen Qwen head.

The objective was:

```text
T² × KL(softmax(teacher / T) || softmax(student / T))
  + 0.1 × MSE(student_logits - mean(student_logits),
              teacher_logits - mean(teacher_logits))
```

with temperature `T=2.0`, AdamW learning rate `1e-3`, weight decay `0.01`, and five epochs. The
permuted control used the same cached outputs but applied a seeded cyclic derangement within groups
having the same candidate count; no example retained its own teacher output.

Runtime checks and focused tests enforce that labeled distillation inputs are rejected, teacher
outputs are detached, no labeled cross-entropy or label-based selection is used, and only the target
adapter can receive gradients. Tensor comparisons verify the source adapter/head and each copied
target head remain unchanged.

The exact command was:

```bash
uv run python scripts/run_distillation_experiment.py \
  --config configs/v0.1.json \
  --train data/pilot-v0.1/train.jsonl \
  --eval data/pilot-v0.1/eval.jsonl \
  --ood data/pilot-v0.1/ood_boolq.jsonl \
  --output runs/diagnostic-label-free-decision-distillation-v0.1-seeds0-2 \
  --device cuda \
  --epochs 5 \
  --temperature 2.0 \
  --centered-logit-mse-weight 0.1 \
  --seeds 0 1 2
```

## Per-seed accuracy

| Seed | Source ID | Native ID | Unaligned ID | Latent ID | Distilled ID | Permuted ID | Distilled OOD | Permuted OOD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7812 | 0.7344 | 0.3281 | 0.4219 | 0.6094 | 0.3438 | 0.5938 | 0.5000 |
| 1 | 0.8281 | 0.7344 | 0.4062 | 0.3281 | 0.7188 | 0.3125 | 0.5938 | 0.5625 |
| 2 | 0.8438 | 0.5781 | 0.2969 | 0.3906 | 0.6562 | 0.3125 | 0.5625 | 0.5000 |

Every per-seed `results.json` contains the complete accuracy, macro-F1, NLL, Brier, ECE,
option-permutation robustness, DPR, training curves, teacher/student metrics, comparisons, and
machine-readable no-label audit.

## Aggregate ID results

Values are mean ± sample standard deviation over seeds 0, 1, and 2. DPR is computed per seed versus
native SmolLM and then aggregated. Option-permutation robustness was 1.0000 ± 0.0000 for every
condition.

| Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | DPR |
|---|---:|---:|---:|---:|---:|---:|
| Qwen source | 0.8177 ± 0.0325 | 0.7960 ± 0.0261 | 1.1962 ± 0.3016 | 0.3467 ± 0.0631 | 0.1752 ± 0.0457 | 1.2170 ± 0.2124 |
| Native SmolLM | 0.6823 ± 0.0902 | 0.6421 ± 0.1168 | 0.9770 ± 0.1326 | 0.4704 ± 0.0909 | 0.2162 ± 0.0531 | 1.0000 ± 0.0000 |
| Supervised DecPort | 0.6927 ± 0.0477 | 0.6613 ± 0.0783 | 0.7998 ± 0.2692 | 0.3992 ± 0.0886 | 0.1629 ± 0.0328 | 1.0295 ± 0.1738 |
| Frozen random head | 0.6823 ± 0.0502 | 0.6616 ± 0.0685 | 0.7760 ± 0.1412 | 0.4434 ± 0.0649 | 0.1614 ± 0.0537 | 1.0153 ± 0.1853 |
| Unaligned + Qwen head | 0.3438 ± 0.0563 | 0.2037 ± 0.0305 | 1.0514 ± 0.0071 | 0.6323 ± 0.0039 | 0.0782 ± 0.0286 | 0.5045 ± 0.0538 |
| Cosine+MSE latent alignment | 0.3802 ± 0.0477 | 0.2358 ± 0.0479 | 1.2384 ± 0.1736 | 0.7215 ± 0.1111 | 0.2507 ± 0.1082 | 0.5657 ± 0.1147 |
| **Decision-space distillation** | **0.6615 ± 0.0549** | **0.6108 ± 0.0595** | 0.9232 ± 0.0942 | 0.4825 ± 0.0695 | 0.1740 ± 0.0747 | **0.9812 ± 0.1527** |
| Permuted-teacher distillation | 0.3229 ± 0.0180 | 0.1670 ± 0.0327 | 2.8410 ± 1.0069 | 1.0728 ± 0.2263 | 0.4933 ± 0.1695 | 0.4781 ± 0.0581 |

Decision distillation gained +0.3177 ± 0.0393 over unaligned, +0.2812 ± 0.1025 over latent
alignment, and +0.3385 ± 0.0705 over the permuted-teacher control.

## Aggregate BoolQ OOD results

| Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | DPR |
|---|---:|---:|---:|---:|---:|---:|
| Qwen source | 0.6146 ± 0.0180 | 0.5317 ± 0.0297 | 1.7041 ± 0.3732 | 0.6022 ± 0.0459 | 0.2816 ± 0.0426 | 1.1362 ± 0.0698 |
| Native SmolLM | 0.5417 ± 0.0180 | 0.3702 ± 0.0403 | 1.7215 ± 0.5270 | 0.7372 ± 0.0806 | 0.3997 ± 0.0276 | 1.0000 ± 0.0000 |
| Supervised DecPort | 0.5417 ± 0.0180 | 0.3513 ± 0.0075 | 1.5488 ± 0.2531 | 0.6863 ± 0.1021 | 0.3581 ± 0.0497 | 1.0000 ± 0.0000 |
| Frozen random head | 0.5417 ± 0.0180 | 0.3702 ± 0.0403 | 1.0823 ± 0.2723 | 0.6033 ± 0.0582 | 0.3005 ± 0.0381 | 1.0011 ± 0.0572 |
| Unaligned + Qwen head | 0.5313 ± 0.0541 | 0.3612 ± 0.0022 | **0.6873 ± 0.0119** | **0.4942 ± 0.0118** | **0.0371 ± 0.0244** | 0.9804 ± 0.0899 |
| Cosine+MSE latent alignment | 0.5729 ± 0.0361 | 0.4120 ± 0.0684 | 0.6908 ± 0.0335 | 0.4970 ± 0.0305 | 0.1503 ± 0.0089 | 1.0577 ± 0.0589 |
| **Decision-space distillation** | **0.5833 ± 0.0180** | 0.3684 ± 0.0072 | 2.3655 ± 0.3816 | 0.7845 ± 0.0389 | 0.4022 ± 0.0114 | **1.0773 ± 0.0349** |
| Permuted-teacher distillation | 0.5208 ± 0.0361 | **0.4716 ± 0.0476** | 0.8492 ± 0.1978 | 0.5715 ± 0.0633 | 0.1722 ± 0.0790 | 0.9630 ± 0.0870 |

Decision distillation gained +0.0521 ± 0.0361 over unaligned, +0.0104 ± 0.0477 over latent
alignment, and +0.0625 ± 0.0312 over the permuted control on OOD accuracy. These small gains do not
offset its much worse calibration and probabilistic loss.

## Teacher/student behavior and optimization

Final behavior is measured on the 128 matching unlabeled training inputs against the correct Qwen
teacher, even for the permuted control.

| Condition | KL | Centered-logit MSE | Top-choice agreement | Probability correlation |
|---|---:|---:|---:|---:|
| Unaligned | 0.7079 ± 0.0611 | 29.2725 ± 7.6732 | 0.3906 ± 0.1228 | 0.3119 ± 0.0419 |
| **Decision-space distillation** | **0.2515 ± 0.1066** | **14.6360 ± 6.7949** | **0.8438 ± 0.0625** | **0.8372 ± 0.0652** |
| Permuted teacher | 0.9793 ± 0.2655 | 28.8175 ± 11.8272 | 0.5182 ± 0.1138 | 0.3376 ± 0.1211 |

The proper objective loss fell from 5.6167 ± 1.3660 at epoch 1 to 2.9559 ± 0.9533 at epoch 5;
temperature-scaled KL before the `T²` multiplier fell from 0.7343 ± 0.1204 to 0.3224 ± 0.0517.
The permuted objective ended at 6.2493 ± 1.0230 with KL 0.8631 ± 0.0110. Complete epoch curves are
in `distillation_summary.json`.

## Reproducibility and provenance

- A clean seed-0 replay reproduced `results.json` exactly after normalizing only `output_dir`; all
  19 retained artifact/config hashes also matched.
- `data_manifest.json` records immutable source splits, counts, preparation command, and SHA-256
  hashes. `environment.json` records Windows, WSL2, CUDA, GPU, Python, and package versions.
- `gpu_usage.csv` contains 1,698 one-second samples. Peak utilization was 65%, peak memory was
  3,260 MiB, peak temperature was 50 C, and peak power was 70.26 W. A rejected initial telemetry
  invocation delayed capture by about 98 seconds; training results were unaffected.
- `distillation_summary.json` contains mean and sample standard deviation for all requested metrics,
  comparison gains, teacher/student behavior, and per-epoch objectives.
- Each seed retains safe source, control, latent-alignment, distilled, and permuted-control
  artifacts. Label-free target artifacts reference the unchanged `source/head.safetensors`.
- `SHA256SUMS` covers every retained result, provenance file, telemetry file, and model artifact.
