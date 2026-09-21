# Architecture

## Overview

DecPort is a Python/PyTorch proof of concept for transferring a learned decision head between frozen
causal language-model backbones. Qwen3-0.6B is the source model and
SmolLM2-360M-Instruct is the first target model.

## Major components

- `src/decport/backbones/`: one small interface plus shared Hugging Face hidden-state extraction.
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

## Data flow

Each candidate option is rendered with the same state and question and encoded independently. The
last non-padding hidden state is mapped by a backbone-specific adapter into a 256-dimensional shared
space by default. The shared head produces one scalar per candidate, and softmax is applied across
the runtime option set.

## External systems

- Hugging Face Transformers supplies Qwen3-0.6B and SmolLM2-360M-Instruct.
- Hugging Face Datasets supplies SST-2, AG News, and the held-out BoolQ family.
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
- Both backbones use the same textual decision prompt.
- Candidate order is shuffled during training and explicitly tested during evaluation.
- Smoke-test metrics are not benchmark evidence.
