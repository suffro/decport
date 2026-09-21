# DecPort

Port decision capabilities across frozen LLM architectures.

DecPort tests whether a small decision head trained with one language model can be reused with a
different frozen language model by training only a lightweight adapter. The initial experiment uses
Qwen3-0.6B as the source and SmolLM2-360M-Instruct as the target.

```text
Qwen3-0.6B (frozen) ── Qwen adapter ──┐
                                      ├── shared decision head ── score
SmolLM2-360M (frozen) ─ Smol adapter ─┘
```

The implementation is working end to end, but there is no credible transfer benchmark yet. DecPort
does not claim that the hypothesis has been validated until the configured experiment is run at a
meaningful scale and its results are independently reproduced.

## Install and test

```bash
uv sync --extra dev --extra train
uv run ruff check .
uv run pytest
```

Real-model inference smoke tests are opt-in because they download both checkpoints:

```bash
DECPORT_RUN_REAL_MODELS=1 uv run pytest tests/test_real_backbones.py
```

## Minimal Choice path

The adapter and head below are untrained; this example demonstrates the API and tensor path only.

```python
from decport import BackboneAdapter, DecisionHead, DecPort, SmolLMBackbone

backbone = SmolLMBackbone.from_pretrained()
model = DecPort(
    backbone,
    BackboneAdapter(backbone.hidden_size),
    DecisionHead(),
)
result = model.choice(
    state="Customer says they were charged twice.",
    question="Which department should handle this?",
    options=["billing", "sales", "technical"],
)
```

`choice()` returns the selected option plus per-option scores and normalized probabilities. Thin
`boolean()` and ordered `score()` wrappers use the same path.

## Reproduce the core experiment

Prepare SST-2 and AG News for training/in-distribution evaluation, with BoolQ held out as an OOD task
family:

```bash
uv run python scripts/prepare_data.py --output data/v0.1
```

Run source training, the native SmolLM baseline, frozen-head transfer, and the frozen random-head
control:

```bash
uv run python scripts/run_experiment.py \
  --config configs/v0.1.json \
  --train data/v0.1/train.jsonl \
  --eval data/v0.1/eval.jsonl \
  --ood data/v0.1/ood_boolq.jsonl \
  --output runs/v0.1
```

The runner writes safetensors artifacts and `results.json` with accuracy, macro-F1, NLL, Brier score,
ECE, option-order robustness, parameter counts, adapter sizes, and the Decision Portability Ratio.
Backbones remain frozen in every run. Transfer freezes the source-trained head; the control freezes a
fresh random head and trains only an identically initialized target adapter. Comparing those two runs
tests whether the source head contributes more than an arbitrary fixed projection target.

## Benchmark status

The first bounded CUDA pilot is archived under
[`benchmarks/pilots/v0.1/`](benchmarks/pilots/v0.1/). A subsequent five-seed, five-epoch
[`convergence diagnostic`](benchmarks/diagnostics/v0.1/) found that the transfer condition's mean
accuracy gain over the matched random-head control narrowed to +0.0063 ± 0.0450 by epoch 5. These
are diagnostic results on small data, not accepted benchmark evidence.

A subsequent bounded
[`label-free latent-alignment diagnostic`](benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-label-free-alignment-seed0/)
trained only a SmolLM adapter to match frozen trained Qwen latents, with all answer fields removed.
It improved ID accuracy over the matched unaligned adapter (0.4219 versus 0.3281) but remained far
below the supervised random-head control (0.7188), so it does not support a portability claim.

A bounded follow-up compared whitening, ridge, and orthogonal Procrustes against that label-free
baseline. None exceeded 0.4219 ID accuracy; see the
[`stronger-method diagnostic`](benchmarks/diagnostics/v0.1/2026-09-21-wsl2-rtx4060ti-label-free-methods-seed0/).

| Backbone | Head | Trainable component | Accuracy | Brier | ECE | Portability Ratio |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | shared/source | adapter + head | pending | pending | pending | — |
| SmolLM2-360M | native | adapter + native head | pending | pending | pending | 1.00 |
| SmolLM2-360M | DecPort | adapter only | pending | pending | pending | pending |
| SmolLM2-360M | random control | adapter only | pending | pending | pending | — |

The `1.00` in the native row is the ratio definition, not an observed result.

## Scope

The v0.1 code intentionally excludes LoRA, serving APIs, UI, custom kernels, vLLM, quantization, and
deployment infrastructure. The priority is a small, falsifiable, reproducible portability experiment.
