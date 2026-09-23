# Architecture

## Overview

DecPort is a Python/PyTorch proof of concept for transferring a learned decision head between frozen
causal language-model backbones. Qwen3-0.6B is the source model; SmolLM2-360M-Instruct, Gemma 3
270M Instruct, and TinyLlama-1.1B-Chat are validated target models.

## Major components

- `src/decport/backbones/`: one small interface, shared Hugging Face hidden-state extraction, Qwen,
  SmolLM, and Gemma wrappers, plus an exact frozen-representation cache for scaled experiments.
- `src/decport/adapter.py`: the only representation-dependent trainable component.
- `src/decport/head.py`: the small shared scalar head.
- `src/decport/model.py`: dynamic Choice scoring and Boolean/basic Score wrappers.
- `src/decport/data.py`: canonical records, JSONL IO, dataset converters, and option shuffling.
- `src/decport/train.py`: source/native or adapter-only training.
- `src/decport/eval.py` and `metrics.py`: ID/OOD and calibration/robustness evaluation.
- `src/decport/serialization.py`: safetensors artifacts.
- `src/decport/experiment.py`: source, native-target, transfer, and random-head control sequence.
- `src/decport/alignment.py`: label-free iterative, whitened, ridge, and Procrustes matching with a
  hard no-answer guard.
- `src/decport/alignment_experiment.py`: bounded source/native/transfer/control/alignment diagnostic.
- `src/decport/distillation.py`: label-free teacher-output collection, decision-space distillation,
  deterministic permuted-teacher control, and teacher/student behavior metrics.
- `src/decport/broad.py`: cached-state minibatch training plus dataset/task/type-stratified metrics
  for Choice, Boolean, and ordered Score decisions.
- `scripts/run_distillation_experiment.py`: controlled multi-seed distillation diagnostic and
  aggregate summary.
- `src/decport/scale_experiment.py` and `scripts/run_scale_experiment.py`: shared-source,
  two-target, five-seed scaled validation with per-target controls and aggregation.
- `src/decport/broad_experiment.py` and `scripts/run_broad_experiment.py`: fixed 4,500/2,000/1,500
  three-target, three-decision-type, five-seed broad validation and recursive aggregation.

## Data flow

Each candidate option is rendered with the same state and question and encoded independently. The
last non-padding hidden state is mapped by a backbone-specific adapter into a 256-dimensional shared
space by default. The shared head produces one scalar per candidate, and softmax is applied across
the runtime option set.

## External systems

- Hugging Face Transformers supplies Qwen3-0.6B, SmolLM2-360M-Instruct, Gemma 3 270M Instruct, and
  TinyLlama-1.1B-Chat.
- Hugging Face Datasets supplies SST-2/AG News/BoolQ for earlier experiments and ARC-Easy, BoolQ,
  Yelp Review Full, OpenBookQA, QNLI, and Amazon Reviews Multi for the broad validation.
- Artifacts use safetensors; no pickle checkpoint format is used.

## Important constraints

- Both LLM backbones stay frozen and in evaluation mode during training.
- No LoRA is used in v0.1.
- The source/shared head is frozen during target transfer.
- The random-head control freezes a fresh head and starts from the same target-adapter weights.
- Label-free alignment freezes both backbones, the trained source adapter/head, and the target copy
  of the source head; only a fresh, matched-initialization target adapter is optimized.
- Alignment inputs must have no `answer`, and the alignment trainer never invokes a decision head.
- Closed-form ridge and Procrustes maps are folded into the target adapter's final linear layer, so
  aligned artifacts retain the standard adapter-to-frozen-head inference path.
- Label-free decision distillation caches detached Qwen candidate scores, uses no answer fields or
  labeled loss, and optimizes only a matched-initialization SmolLM adapter through the unchanged
  frozen Qwen head.
- Direct logit matching is invariant to arbitrary per-decision offsets because logits are centered
  before MSE. The KL term uses temperature-softened candidate distributions.
- All backbones use the same textual decision prompt.
- In the scaled two-target experiment, the exact same trained Qwen head artifact is used by both
  targets within each seed and verified by SHA-256.
- In the broad experiment, the exact same trained Qwen head is reused by all three targets within
  each seed; all six target conditions start from one target-specific adapter state, and the strict
  label-free entry point rejects labeled records.
- Frozen hidden-state caching is permitted as an exact execution optimization: it must not change
  prompts, per-decision optimization steps, losses, trainable components, or evaluation behavior.
- Candidate order is shuffled during training and explicitly tested during evaluation.
- Smoke-test metrics are not benchmark evidence.
