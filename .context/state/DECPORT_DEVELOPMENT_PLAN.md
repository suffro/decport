# DecPort — Development Plan

## 1. Goal

DecPort should prove one central idea as quickly and convincingly as possible:

> **Train a decision head once. Port it to another LLM by training only a lightweight adapter.**

The first release should be a strong, reproducible proof-of-concept rather than a complete production framework.

The main value proposition is **cross-backbone portability of decision capabilities**, not simply another Jev/System-One clone.

---

## 2. Core v0.1 Hypothesis

A decision capability learned on one language-model backbone can be reused on a different backbone if the target model's hidden representations are mapped into a shared decision space through a lightweight adapter.

Target architecture:

```text
                SHARED FROZEN
                DECISION HEAD
                     │
          ┌──────────┴──────────┐
          │                     │
   Qwen projector        SmolLM projector
          │                     │
     Qwen3-0.6B          SmolLM2-360M
          │                     │
      frozen LLM            frozen LLM
```

The key experiment is:

1. Train the source adapter + shared decision head on Qwen.
2. Freeze the shared decision head.
3. Train only a lightweight adapter for SmolLM.
4. Compare the transferred setup against a native SmolLM-specific head.

If the transferred setup retains most of the native target performance, DecPort has demonstrated its core value.

---

## 3. Initial Backbones

### Source backbone

**Qwen3-0.6B**

Reasons:

- small enough for fast iteration;
- modern architecture;
- strong open ecosystem;
- suitable for local or inexpensive cloud experimentation.

### Target backbone

**SmolLM2-360M-Instruct**

Reasons:

- very small;
- different model family;
- cheap and fast to train;
- useful for proving that DecPort is genuinely cross-backbone.

### Third backbone

Add only after the first transfer works.

Candidates:

- Gemma small model;
- another Llama-family model;
- Phi-family model.

Two models prove feasibility. Three models make the portability claim much stronger.

---

## 4. Minimal Architecture

For v0.1, keep the architecture deliberately simple.

```text
STATE
+
QUESTION
+
CANDIDATE OPTION
       ↓
     LLM
       ↓
hidden representation
       ↓
DecPort adapter
       ↓
shared decision space
       ↓
shared decision head
       ↓
scalar score
```

For a Choice task:

```text
Question:
Which department should handle this?

Options:
- billing
- sales
- technical
```

Each option receives a scalar score:

```text
billing   →  2.91
sales     → -0.83
technical →  0.44
```

Then:

```text
softmax
↓
billing   0.89
sales     0.02
technical 0.09
```

This preserves dynamic runtime options rather than requiring a fixed classifier label set.

---

## 5. Decision Primitives

### v0.1

Implement:

- **Choice**
- **Boolean / Noul**
- **basic Score**

### Mapping

Boolean can be represented as:

```text
yes / no
```

Score can initially be represented through ordered options:

```text
very_low
low
medium
high
very_high
```

with an expected-value calculation layered on top.

Do not build sophisticated continuous scoring before the transfer hypothesis is validated.

---

## 6. Adapter Design

Start with the smallest architecture likely to work.

Example:

```text
hidden state
    ↓
LayerNorm
    ↓
Linear(d_model → 256)
    ↓
GELU
    ↓
Linear(256 → 256)
```

Example mappings:

```text
Qwen    → 256-dimensional shared space
SmolLM  → 256-dimensional shared space
```

The backbone-specific adapter converts each model's native representation into the common DecPort decision space.

### Shared head

Keep the shared head very small:

```text
256
 ↓
LayerNorm
 ↓
Linear(256 → 1)
 ↓
score
```

### Important constraint

For the first experiment:

> **Keep the LLM backbone frozen.**

Do not use LoRA unless the frozen-backbone experiment fails badly.

If DecPort works with frozen backbones, the result is substantially more interesting.

---

## 7. Core Experiment

### Experiment A — Source model

```text
Frozen Qwen
     ↓
Qwen adapter
     ↓
Shared decision head
```

Train:

- Qwen adapter;
- shared decision head.

Then freeze the shared decision head permanently.

---

### Experiment B — Native target baseline

```text
Frozen SmolLM
     ↓
SmolLM adapter
     ↓
SmolLM-specific decision head
```

Train both target components.

This establishes the best result achievable by the target model under the same general setup.

---

### Experiment C — DecPort transfer

```text
Frozen SmolLM
     ↓
TRAINABLE SmolLM adapter
     ↓
FROZEN Qwen-trained shared head
```

Only the SmolLM adapter is trained.

This is the central DecPort experiment.

The adapter must learn how to map SmolLM representations into the decision space already understood by the frozen shared head.

---

## 8. Primary Success Metric

Introduce a simple project-specific metric:

## Decision Portability Ratio

```text
DecPort target performance
──────────────────────────
Native target performance
```

Example:

```text
Native SmolLM head:       81.2%
DecPort shared head:      78.9%

Portability ratio:        97.2%
```

### Suggested v0.1 success threshold

A reasonable first target:

```text
DecPort target ≥ 90% of native target performance
```

This does **not** mean 90% raw accuracy.

Example:

```text
Native target = 80%
DecPort       = 72%

Portability ratio = 90%
```

A much stronger outcome would be 95–100%.

---

## 9. Evaluation Metrics

Every release benchmark should include:

| Metric | Purpose |
|---|---|
| Accuracy | basic task performance |
| Macro F1 | robustness to class imbalance |
| NLL | probabilistic quality |
| Brier score | probability calibration |
| ECE | calibration error |
| Decision Portability Ratio | DecPort-specific transfer quality |
| Trainable parameters | transfer efficiency |
| Adapter size | deployment cost |
| Option permutation robustness | detect positional shortcuts |

Also include at least one out-of-distribution evaluation.

---

## 10. Dataset Strategy

Do not build a huge proprietary dataset before validating the architecture.

Convert existing classification/decision datasets into a unified format:

```json
{
  "state": "...",
  "question": "...",
  "options": ["...", "...", "..."],
  "answer": "..."
}
```

Useful task families:

- sentiment;
- topic classification;
- intent routing;
- entailment;
- boolean QA;
- multi-class semantic decisions.

### Critical rule

Shuffle option ordering continuously during training.

```text
shuffle(options)
```

This reduces the risk that the model learns positional shortcuts.

### OOD split

Keep at least one entire task family out of training.

Example:

- train on sentiment, routing, topic, entailment;
- evaluate OOD on a separate semantic-decision task.

---

## 11. High-Value Follow-up Experiment

Do not block v0.1 on this, but prioritize it immediately afterward.

Instead of training the target adapter using decision labels, align the hidden spaces directly.

Conceptually:

```text
Qwen hidden representation
        ↓
Qwen projector
        ↓
shared representation

SmolLM hidden representation
        ↓
SmolLM projector
        ↓
shared representation
```

Optimize:

```text
P_target(h_target) ≈ P_source(h_source)
```

Possible losses:

- cosine similarity;
- MSE;
- contrastive loss;
- combinations of the above.

Then test:

```text
Target LLM
   ↓
adapter trained without decision labels
   ↓
frozen source decision head
```

If this works, DecPort gains a much stronger claim:

> **Port a learned decision capability to another LLM without retraining it on the original decision task.**

---

## 12. v0.1 Scope

### Must ship

- Qwen source model;
- SmolLM target model;
- frozen shared decision head;
- lightweight per-backbone adapters;
- dynamic Choice;
- Boolean/Noul wrapper;
- basic Score wrapper;
- reproducible training scripts;
- native target baseline;
- transfer benchmark;
- calibration metrics;
- option-order robustness test;
- adapter serialization;
- unit tests;
- clean Python API;
- Hugging Face checkpoints/adapters;
- strong README;
- architecture diagram.

### Explicitly postpone

- REST API;
- Jev-compatible server API;
- web UI;
- playground;
- custom CUDA kernels;
- block-causal attention;
- multi-question one-pass optimization;
- vLLM integration;
- quantization work;
- production serving;
- support for many model families;
- complex LoRA strategies;
- custom training infrastructure.

The release should demonstrate the research result, not reproduce the whole Jev ecosystem.

---

## 13. Repository Structure

Suggested initial layout:

```text
decport/
├── src/
│   └── decport/
│       ├── backbones/
│       │   ├── base.py
│       │   ├── qwen.py
│       │   └── smollm.py
│       ├── adapter.py
│       ├── head.py
│       ├── model.py
│       ├── schema.py
│       ├── train.py
│       ├── eval.py
│       └── metrics.py
│
├── configs/
├── benchmarks/
├── examples/
├── scripts/
├── tests/
├── README.md
├── pyproject.toml
├── LICENSE
└── CHANGELOG.md
```

Keep dependencies minimal:

- PyTorch;
- Transformers;
- datasets;
- safetensors;
- standard evaluation utilities.

---

## 14. Backbone Interface

Create one small abstraction early.

Example conceptual interface:

```python
class DecPortBackbone:
    hidden_size: int

    def encode(self, state, question, option):
        ...
```

Backbone-specific implementations:

```text
QwenBackbone
SmolLMBackbone
GemmaBackbone
LlamaBackbone
```

Everything above this layer should remain backbone-agnostic.

This is important because the real product value of DecPort comes from adding new models cheaply.

---

## 15. Python API

The v0.1 API should already feel like a real library.

Example:

```python
from decport import DecPort

model = DecPort.from_pretrained(
    "suffro/decport-smollm2-360m"
)

result = model.choice(
    state="Customer says they were charged twice.",
    question="Which department should handle this?",
    options=["billing", "sales", "technical"],
)

print(result)
```

Expected output:

```python
{
    "choice": "billing",
    "probabilities": {
        "billing": 0.91,
        "sales": 0.02,
        "technical": 0.07
    }
}
```

Do not over-engineer the API before this basic flow works.

---

## 16. Benchmark Presentation

The README should contain one central table.

Example structure:

| Backbone | Head | Trainable component | Accuracy | Brier | ECE | Portability Ratio |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | shared/source | adapter + head | — | — | — | — |
| SmolLM2-360M | native | adapter + native head | — | — | — | 1.00 |
| SmolLM2-360M | DecPort | adapter only | — | — | — | — |

The most important comparison is:

```text
native target
vs
DecPort target with frozen source head
```

---

## 17. Release Strategy

Release GitHub and Hugging Face together.

### GitHub

Ship:

- library;
- training scripts;
- evaluation code;
- configs;
- benchmarks;
- reproducibility instructions;
- architecture diagram.

### Hugging Face

Publish:

- source adapter;
- shared decision head;
- target DecPort adapter;
- model cards;
- benchmark results.

### Package

If packaging is ready:

```bash
pip install decport
```

Otherwise initially support:

```bash
uv add git+https://github.com/<org>/decport
```

PyPI should not delay the first public release.

---

## 18. README Structure

The top of the README should communicate the project in seconds.

```text
DecPort
Port decision capabilities across LLM architectures.

[architecture diagram]

Train the decision head once.
Adapt another backbone with a lightweight projector.

Qwen3-0.6B → SmolLM2-360M

[benchmark table]
```

Then immediately provide:

1. one-paragraph explanation;
2. installation;
3. 10-line quickstart;
4. benchmark;
5. architecture;
6. reproducibility;
7. roadmap.

Avoid a long theoretical introduction before showing the result.

---

## 19. Exact Development Order

### Phase 0 — Skeleton

1. create repository;
2. add `pyproject.toml`;
3. establish package structure;
4. add CI for lint/tests;
5. define unified decision schema.

### Phase 1 — Core inference

6. generic backbone interface;
7. Qwen hidden-state extraction;
8. SmolLM hidden-state extraction;
9. adapter implementation;
10. shared scalar decision head;
11. Choice inference;
12. option softmax.

### Phase 2 — Data

13. unified dataset format;
14. converters for a small set of datasets;
15. dynamic option shuffling;
16. train/validation/OOD splits.

### Phase 3 — Source training

17. freeze Qwen;
18. train Qwen adapter + shared head;
19. save source head;
20. benchmark source model.

### Phase 4 — Baselines

21. freeze SmolLM;
22. train SmolLM adapter + native head;
23. benchmark native target.

### Phase 5 — DecPort experiment

24. freeze source shared head;
25. attach SmolLM adapter;
26. train only SmolLM adapter;
27. benchmark transferred target;
28. calculate Decision Portability Ratio.

### Phase 6 — Robustness

29. option-order permutation tests;
30. Brier score;
31. ECE;
32. NLL;
33. OOD evaluation;
34. trainable-parameter and adapter-size reporting.

### Phase 7 — Library polish

35. Boolean/Noul wrapper;
36. Score wrapper;
37. serialization;
38. `from_pretrained`;
39. simple CLI if trivial;
40. unit tests.

### Phase 8 — Public release

41. architecture diagram;
42. benchmark table;
43. model cards;
44. README;
45. Hugging Face uploads;
46. GitHub release `v0.1.0`.

### Phase 9 — Immediate follow-up

47. add third backbone;
48. latent-alignment experiment;
49. investigate label-free target adaptation;
50. compare multiple adapter architectures.

---

## 20. Release Gate

Do not wait for DecPort to be feature-complete.

Release as soon as all of these are true:

- source training is reproducible;
- target native baseline exists;
- transferred target result exists;
- frozen-head transfer is measurably functional;
- benchmark scripts are public;
- results can be independently reproduced;
- Python quickstart works;
- README explains the contribution clearly.

If transfer performance is weak, fix the adapter/alignment before making strong portability claims.

Do not hide negative results.

---

## 21. What Not to Optimize Yet

Avoid spending early development time on:

- inference micro-optimizations;
- custom kernels;
- UI;
- hosted services;
- elaborate branding;
- broad compatibility matrices;
- ten different backbones;
- distributed training;
- complex deployment targets.

The highest-value artifact is a credible experimental result.

---

## 22. v0.2 Priorities

After v0.1:

1. third backbone;
2. direct latent-space alignment;
3. adapter-only transfer without task labels;
4. better calibration;
5. abstention / uncertainty thresholds;
6. batch and multi-question inference;
7. more adapter architectures;
8. LoRA fallback for difficult backbones;
9. standardized benchmark suite;
10. broader Hugging Face integration.

---

## 23. v0.3+ Direction

Only after cross-backbone transfer is established:

- universal/shared latent decision space across several model families;
- adapter registry;
- automated `decport adapt <model>` workflow;
- Jev-compatible API layer;
- optimized parallel decision inference;
- quantized adapters;
- vLLM / SGLang integration;
- production routing;
- confidence-based escalation to larger reasoning models;
- benchmark leaderboard.

Longer-term target:

```bash
decport adapt Qwen/Qwen3-0.6B
decport adapt HuggingFaceTB/SmolLM2-360M-Instruct
decport adapt google/gemma-...
```

with each new model learning only a lightweight mapping into an existing decision space.

---

## 24. Core Technical Claim

The project should stay centered on a falsifiable claim:

> **Decision capabilities represented by a learned head can be transferred across heterogeneous LLM backbones through lightweight latent-space adapters while retaining most of the target model's native decision performance.**

Everything in v0.1 should exist to test, measure, reproduce, or demonstrate that claim.

---

## 25. Positioning

Do **not** position DecPort as:

> another open-source Jev clone.

Position it as:

> **A cross-backbone portability layer for learned LLM decision capabilities.**

Short version:

> **Train the decision head once. Port it across LLMs.**

That is the distinction DecPort should prove first.
