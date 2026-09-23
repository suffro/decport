# DecPort

Port decision capabilities across frozen LLM architectures.

> **Final verdict: NO-SHIP.**
>
> - The pre-registered final gate (decision 0007) tested whether one learned decision core, frozen
>   and reused through lightweight adapters, is useful across SmolLM2-360M, Gemma 3 270M, and
>   TinyLlama-1.1B. It is not.
> - The learned core is indistinguishable from a scale-matched random core, and a small
>   target-specific distilled module does as well.
> - Label-free decision-behavior distillation itself works in-distribution, but it does not need a
>   shared core.
> - No v0.1 release is made. See the
>   [final report](benchmarks/accepted/decport-final-gate-v0.1/FINAL_REPORT.md).

DecPort tests whether a small decision head trained with one language model can be reused with a
different frozen language model by training only a lightweight adapter. Current validation uses
Qwen3-0.6B as source and SmolLM2-360M, Gemma 3 270M, and TinyLlama-1.1B as targets.

```text
Qwen3-0.6B (frozen) ── Qwen adapter ──┐
                                      ├── shared decision head ── score
SmolLM2-360M (frozen) ─ Smol adapter ─┘
```

The implementation is working end to end. Five-seed evidence now supports label-free transfer for
Choice and ordered Score across three heterogeneous target families, while Boolean transfer remains
unresolved. This is diagnostic evidence rather than a universal portability claim.

## Install and test

```bash
uv sync --extra dev --extra train --extra openjev
uv run ruff check .
uv run pytest
```

The `openjev` extra installs Open-Jev at a pinned commit, plus PEFT and Accelerate, for the external
teacher experiment below.

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

A three-seed
[`decision-space distillation diagnostic`](benchmarks/diagnostics/v0.1/2026-09-22-wsl2-rtx4060ti-decision-distillation-seeds0-2/)
then trained only the SmolLM adapter against frozen Qwen candidate distributions. Mean ID accuracy
was 0.6615 ± 0.0549, compared with 0.3802 ± 0.0477 for latent alignment and 0.3229 ± 0.0180 for a
permuted-teacher control. This is positive bounded evidence for transferring input-specific decision
behavior, but weak OOD gains and degraded calibration preclude a general portability claim.

The
[`broad Jev-like validation`](benchmarks/diagnostics/v0.1/2026-09-23-wsl2-rtx4060ti-jev-broad-seeds0-4/)
uses 4,500 train, 2,000 ID, and 1,500 OOD examples across Choice, Boolean, and ordered Score, with
five seeds and three heterogeneous targets. Correct Qwen behavior beats untrained and mismatched
teacher controls overall for every target. Choice and Score transfer across all targets ID, and
Score remains positive under dataset shift; Boolean does not reliably beat the mismatched control.

| Backbone | Head | Trainable component | Accuracy | Brier | ECE | Portability Ratio |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | shared/source | adapter + head | pending | pending | pending | — |
| SmolLM2-360M | native | adapter + native head | pending | pending | pending | 1.00 |
| SmolLM2-360M | DecPort | adapter only | pending | pending | pending | pending |
| SmolLM2-360M | random control | adapter only | pending | pending | pending | — |

The `1.00` in the native row is the ratio definition, not an observed result.

## Open-Jev DecisionCore transfer (accepted, decision 0006)

This experiment asks one narrow question: can the decision behavior of a real pretrained
[Open-Jev](https://github.com/Zefan-Cai/Open-Jev) model move to heterogeneous frozen backbones
through DecPort adapters, while the same Open-Jev decision head and calibration stay frozen?

```text
Open-Jev 2B (Qwen3.5-2B + LoRA, frozen) ── last token (2048) ──┐  teacher distributions
                                                                ▼
                     ONE frozen Open-Jev DecisionCore: trained Linear(2048→1) + typed assembly + T
                                                                ▲
SmolLM2 / Gemma 3 270M / TinyLlama (frozen) ── adapter d→256→2048 (only trainable part)
```

- The teacher is the released `ZefanCai/Open-Jev-2B`, loaded by Open-Jev's own loader at pinned
  revisions. Its trained head and saved temperature are the frozen core. Every teacher batch
  proves the core reproduces the upstream head exactly.
- Choice, Noul, and Score keep Open-Jev semantics. Noul is one prompt with logits `[0, s]`, not a
  two-option Choice. Metrics are reported per type.
- Adapters learn only from Open-Jev's calibrated distributions on unlabeled inputs. Controls: the
  untrained adapter, a deterministic mismatched teacher, and a random frozen core.
- [JevBench](https://github.com/fstandhartinger/jevbench) runs through a thin adapter over a clean
  pinned checkout. Only the 231-task public subset exists locally; results are public-subset only.

```bash
# Multi-seed experiment on the broad Jev-like data (see data/jev-broad-v0.1 preparation above).
bash scripts/run_openjev_wsl_archive.sh data/jev-broad-v0.1 runs/openjev-transfer-v0.1-seeds0-4

# Public JevBench subset for the teacher or a ported target.
git clone https://github.com/fstandhartinger/jevbench ../jevbench
git -C ../jevbench checkout 2fa63fa3226cb369795525ed011800f57dcbd894
uv run python scripts/run_jevbench_public.py --jevbench-root ../jevbench \
  --system openjev-teacher --output runs/jevbench-openjev-teacher
```

The five-seed run is accepted and archived under
[`benchmarks/accepted/openjev-transfer-v0.1/`](benchmarks/accepted/openjev-transfer-v0.1/). Behavior
transfers on Choice and Score, but a random frozen core does as well as Open-Jev's rank-one head,
and Noul fails.

## Final ship gate (decision 0007)

The final gate replaced the rank-one head with a nonlinear DecisionCore, learned once with labels
on source-only Open-Jev representations and then frozen. It also replaced the expressive adapter
with a rank-128 linear map, and added a scale-matched random core and a target-specific distilled
module as controls. The verdict is **NO-SHIP**; see the
[final report](benchmarks/accepted/decport-final-gate-v0.1/FINAL_REPORT.md) and
[archive](benchmarks/accepted/decport-final-gate-v0.1/2026-09-23-wsl2-rtx4060ti-seeds0-4/).

```bash
uv run python scripts/prepare_source_core_data.py --output data/jev-broad-v0.1-source-core \
  --broad-data data/jev-broad-v0.1
bash scripts/run_final_gate_wsl_archive.sh data/jev-broad-v0.1 data/jev-broad-v0.1-source-core \
  runs/decport-final-gate-v0.1-seeds0-4
```

## Scope

The v0.1 code intentionally excludes LoRA, serving APIs, UI, custom kernels, vLLM, quantization, and
deployment infrastructure. The priority is a small, falsifiable, reproducible portability experiment.
