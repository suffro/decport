# v0.1 scaled cross-backbone decision-transfer diagnostic

## Status and answer

Completed successfully on 2026-09-23. This is a five-seed diagnostic on 1,280 training, 640
in-distribution evaluation, and 320 BoolQ OOD examples: exactly 10 times every split in the prior
128/64/32 diagnostic. The train, ID, and OOD records are disjoint, and seeds 0–4 use one exact
frozen Qwen decision head for both target backbones within each seed.

The requested ID pattern reproduced at scale and on both targets:

- SmolLM: distillation reached 0.7828 ± 0.0173 accuracy, versus 0.4078 ± 0.0169 unaligned and
  0.3713 ± 0.0130 with the permuted teacher.
- Gemma: distillation reached 0.7888 ± 0.0295 accuracy, versus 0.3938 ± 0.0266 unaligned and
  0.3813 ± 0.0301 with the permuted teacher.

Correct-teacher gains were +0.3750 ± 0.0173 over unaligned and +0.4116 ± 0.0225 over permuted for
SmolLM, and +0.3950 ± 0.0431 over unaligned and +0.4075 ± 0.0388 over permuted for Gemma. The
effect held in every seed. This is stronger evidence for input-specific cross-backbone decision
transfer through the same frozen Qwen head without target labels. It is not a general portability
claim: only two target families and two ID task families have been tested, and BoolQ OOD gains were
small and noisy with worse calibration.

## Protocol

- Source: `Qwen/Qwen3-0.6B`.
- Targets: `HuggingFaceTB/SmolLM2-360M-Instruct` and the ungated safetensors mirror
  `unsloth/gemma-3-270m-it` of Google's Gemma 3 270M instruction checkpoint.
- Shared latent size 256; maximum sequence length 512.
- Five fixed epochs for every trained condition; AdamW learning rate `1e-3`, weight decay `0.01`.
- Decision distillation uses the unchanged `T=2.0` objective with centered-logit MSE weight `0.1`.
- No target- or seed-specific hyperparameter tuning and no early stopping/model selection.
- Frozen backbone hidden states were batch-cached once and reused exactly. Adapter optimization
  retained the original per-decision update sequence and objective.

For each seed, the Qwen adapter/head was trained once, frozen, and its head artifact checksum was
referenced by both targets. Each target's six conditions began from the same target-adapter state:
unaligned, correct-teacher distillation, permuted-teacher distillation, native supervised,
supervised DecPort, and frozen random-head supervised control.

All distillation inputs were reconstructed with `answer=None`. Runtime guards rejected labeled
records, cached detached teacher scores, froze the Qwen backbone/adapter/head and target backbone,
froze the copied Qwen head, and optimized only the target adapter. The audit script verified these
properties, all artifact layouts and head checksums, all dataset hashes, and finite recorded values.

## Aggregate ID results

Values are mean ± sample standard deviation across seeds 0–4. DPR is relative to the matching
native target in each seed. Option-permutation robustness was 1.0000 ± 0.0000 throughout.

| Target | Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | DPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | source/reference ceiling | 0.8341 ± 0.0272 | 0.8412 ± 0.0215 | 0.5799 ± 0.1716 | 0.2590 ± 0.0492 | 0.0883 ± 0.0418 | — |
| SmolLM | unaligned + Qwen head | 0.4078 ± 0.0169 | 0.2747 ± 0.0425 | 1.0379 ± 0.0095 | 0.6249 ± 0.0066 | 0.0483 ± 0.0289 | 0.4934 ± 0.0243 |
| SmolLM | **decision distillation** | **0.7828 ± 0.0173** | **0.7975 ± 0.0160** | 0.7304 ± 0.2680 | **0.3379 ± 0.0390** | 0.1124 ± 0.0374 | **0.9468 ± 0.0191** |
| SmolLM | permuted teacher | 0.3713 ± 0.0130 | 0.2128 ± 0.0211 | 1.2086 ± 0.2252 | 0.7195 ± 0.1080 | 0.1849 ± 0.1123 | 0.4489 ± 0.0128 |
| SmolLM | native supervised | 0.8269 ± 0.0118 | 0.8296 ± 0.0129 | 0.6861 ± 0.0330 | 0.2809 ± 0.0159 | 0.1110 ± 0.0070 | 1.0000 ± 0.0000 |
| SmolLM | supervised DecPort | 0.8181 ± 0.0113 | 0.8228 ± 0.0113 | 0.7273 ± 0.0836 | 0.2944 ± 0.0220 | 0.1184 ± 0.0096 | 0.9894 ± 0.0076 |
| SmolLM | frozen random head | 0.8013 ± 0.0229 | 0.8086 ± 0.0228 | **0.5632 ± 0.0986** | 0.3024 ± 0.0392 | **0.0650 ± 0.0281** | 0.9689 ± 0.0145 |
| Gemma | unaligned + Qwen head | 0.3938 ± 0.0266 | 0.2627 ± 0.0651 | 1.0472 ± 0.0184 | 0.6302 ± 0.0087 | 0.0639 ± 0.0343 | 0.4885 ± 0.0322 |
| Gemma | **decision distillation** | **0.7888 ± 0.0295** | **0.8029 ± 0.0271** | **0.5972 ± 0.1713** | **0.3111 ± 0.0563** | **0.1020 ± 0.0449** | **0.9786 ± 0.0404** |
| Gemma | permuted teacher | 0.3813 ± 0.0301 | 0.2255 ± 0.0398 | 1.1896 ± 0.2369 | 0.7115 ± 0.1116 | 0.1835 ± 0.1130 | 0.4742 ± 0.0545 |
| Gemma | native supervised | 0.8066 ± 0.0311 | 0.8195 ± 0.0255 | 0.6967 ± 0.1308 | 0.3120 ± 0.0487 | 0.1248 ± 0.0251 | 1.0000 ± 0.0000 |
| Gemma | supervised DecPort | 0.8197 ± 0.0129 | 0.8288 ± 0.0122 | 0.6442 ± 0.1142 | 0.2876 ± 0.0262 | 0.1124 ± 0.0228 | 1.0171 ± 0.0289 |
| Gemma | frozen random head | 0.7903 ± 0.0416 | 0.8044 ± 0.0395 | 0.7358 ± 0.1358 | 0.3513 ± 0.0583 | 0.1429 ± 0.0319 | 0.9799 ± 0.0365 |

## Aggregate BoolQ OOD results

| Target | Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | DPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | source/reference | 0.6725 ± 0.0046 | 0.6550 ± 0.0031 | 1.1350 ± 0.1305 | 0.5512 ± 0.0152 | 0.2351 ± 0.0261 | — |
| SmolLM | unaligned + Qwen head | 0.5350 ± 0.0507 | 0.4646 ± 0.0344 | 0.6897 ± 0.0095 | 0.4965 ± 0.0095 | 0.0391 ± 0.0169 | 0.8787 ± 0.0809 |
| SmolLM | decision distillation | 0.6013 ± 0.0017 | 0.3827 ± 0.0007 | 3.3318 ± 1.0748 | 0.7749 ± 0.0071 | 0.3883 ± 0.0045 | 0.9877 ± 0.0085 |
| SmolLM | permuted teacher | 0.5363 ± 0.0886 | 0.3845 ± 0.0591 | 0.6985 ± 0.0307 | 0.5046 ± 0.0301 | 0.0769 ± 0.0619 | 0.8806 ± 0.1433 |
| SmolLM | native supervised | 0.6088 ± 0.0041 | 0.3957 ± 0.0095 | 4.3203 ± 0.6786 | 0.7661 ± 0.0069 | 0.3817 ± 0.0046 | 1.0000 ± 0.0000 |
| SmolLM | supervised DecPort | 0.6069 ± 0.0026 | 0.3907 ± 0.0035 | 4.1551 ± 0.2815 | 0.7621 ± 0.0035 | 0.3810 ± 0.0025 | 0.9970 ± 0.0078 |
| SmolLM | frozen random head | 0.6013 ± 0.0017 | 0.3784 ± 0.0046 | 1.9572 ± 0.2728 | 0.6637 ± 0.0184 | 0.2850 ± 0.0323 | 0.9877 ± 0.0058 |
| Gemma | unaligned + Qwen head | 0.5481 ± 0.0828 | 0.4135 ± 0.0558 | 0.6910 ± 0.0201 | 0.4978 ± 0.0200 | 0.0599 ± 0.0508 | 1.0065 ± 0.1479 |
| Gemma | decision distillation | 0.5631 ± 0.0149 | 0.4535 ± 0.0387 | 1.6462 ± 0.4192 | 0.7170 ± 0.0289 | 0.3283 ± 0.0278 | 1.0346 ± 0.0305 |
| Gemma | permuted teacher | 0.5144 ± 0.0906 | 0.4348 ± 0.0915 | 0.7099 ± 0.0280 | 0.5147 ± 0.0281 | 0.0868 ± 0.0693 | 0.9439 ± 0.1592 |
| Gemma | native supervised | 0.5444 ± 0.0068 | 0.5190 ± 0.0178 | 1.8643 ± 0.2555 | 0.7557 ± 0.0138 | 0.3390 ± 0.0109 | 1.0000 ± 0.0000 |
| Gemma | supervised DecPort | 0.5456 ± 0.0098 | 0.5311 ± 0.0165 | 1.3300 ± 0.1392 | 0.6838 ± 0.0173 | 0.2635 ± 0.0180 | 1.0024 ± 0.0201 |
| Gemma | frozen random head | 0.5175 ± 0.0252 | 0.4942 ± 0.0114 | 0.8287 ± 0.0564 | 0.5607 ± 0.0243 | 0.1327 ± 0.0292 | 0.9510 ± 0.0535 |

Correct-teacher OOD gains were +0.0663 ± 0.0514 over unaligned and +0.0650 ± 0.0893 over
permuted for SmolLM, and +0.0150 ± 0.0913 and +0.0488 ± 0.0961 for Gemma. These do not establish
OOD transfer. The much worse NLL/Brier/ECE after distillation confirms the prior calibration concern.

## Teacher/student decision behavior on ID evaluation

All values compare each target to the matching frozen Qwen teacher on the 640 ID inputs after
answers were removed.

| Target | Condition | Top-choice agreement | Probability correlation | KL divergence |
|---|---|---:|---:|---:|
| SmolLM | unaligned | 0.4191 ± 0.0556 | 0.3182 ± 0.0258 | 0.6397 ± 0.0769 |
| SmolLM | **decision distillation** | **0.8169 ± 0.0234** | **0.8533 ± 0.0115** | **0.2213 ± 0.0559** |
| SmolLM | permuted teacher | 0.4278 ± 0.0476 | 0.3410 ± 0.0334 | 0.6379 ± 0.0832 |
| Gemma | unaligned | 0.4038 ± 0.0409 | 0.3160 ± 0.0198 | 0.6415 ± 0.0788 |
| Gemma | **decision distillation** | **0.8113 ± 0.0249** | **0.8565 ± 0.0247** | **0.2003 ± 0.0478** |
| Gemma | permuted teacher | 0.4316 ± 0.0673 | 0.3522 ± 0.0220 | 0.6293 ± 0.0793 |

Per-seed `results.json` files also contain train and OOD behavior metrics for every condition.

## Per-seed ID accuracy

| Target | Seed | Unaligned | Distilled | Permuted | Native | Supervised DecPort | Random head |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM | 0 | 0.3875 | 0.7828 | 0.3750 | 0.8344 | 0.8234 | 0.8203 |
| SmolLM | 1 | 0.4203 | 0.7703 | 0.3516 | 0.8125 | 0.8000 | 0.7703 |
| SmolLM | 2 | 0.4250 | 0.8109 | 0.3688 | 0.8375 | 0.8219 | 0.8219 |
| SmolLM | 3 | 0.4141 | 0.7828 | 0.3734 | 0.8156 | 0.8156 | 0.7844 |
| SmolLM | 4 | 0.3922 | 0.7672 | 0.3875 | 0.8344 | 0.8297 | 0.8094 |
| Gemma | 0 | 0.4109 | 0.7734 | 0.3766 | 0.7891 | 0.8141 | 0.8188 |
| Gemma | 1 | 0.3766 | 0.7891 | 0.4328 | 0.7641 | 0.8094 | 0.7328 |
| Gemma | 2 | 0.4203 | 0.8234 | 0.3672 | 0.8453 | 0.8406 | 0.8406 |
| Gemma | 3 | 0.3563 | 0.8094 | 0.3750 | 0.8203 | 0.8109 | 0.7844 |
| Gemma | 4 | 0.4047 | 0.7484 | 0.3547 | 0.8141 | 0.8234 | 0.7750 |

## Reproduction and archive

Data preparation:

```bash
uv run python scripts/prepare_data.py \
  --output data/decision-transfer-scale-v0.1 \
  --train-per-family 640 --eval-per-family 320 --ood 320 --seed 1729
```

Experiment:

```bash
uv run python scripts/run_scale_experiment.py \
  --config configs/decision-transfer-scale-v0.1.json \
  --train data/decision-transfer-scale-v0.1/train.jsonl \
  --eval data/decision-transfer-scale-v0.1/eval.jsonl \
  --ood data/decision-transfer-scale-v0.1/ood_boolq.jsonl \
  --output runs/decision-transfer-scale-v0.1-seeds0-4
```

Audit:

```bash
uv run python scripts/audit_scale_results.py \
  --run runs/decision-transfer-scale-v0.1-seeds0-4 \
  --data data/decision-transfer-scale-v0.1
```

The audit passed. `aggregate_summary.json` contains every mean and sample standard deviation;
`seed-*/results.json` contains complete per-seed records; `experiment_config.json` and
`repository_config.json` retain configuration; `data_manifest.json` retains source fingerprints,
counts, and split hashes; `environment.json` pins model revisions and software/hardware provenance;
and `SHA256SUMS` covers every retained file except itself.

The run lasted from 2026-09-22 19:02:27 to 2026-09-23 01:13:50 Europe/Rome. Telemetry contains
20,614 samples; peak GPU memory was 7,888 MiB, utilization 100%, temperature 57 C, and power
133.48 W. Gemma caching emitted one recoverable allocator warning when its attention implementation
attempted an oversized temporary allocation; it fell back and the run completed all seeds with
finite results. No hyperparameter, condition, or seed was changed in response.
