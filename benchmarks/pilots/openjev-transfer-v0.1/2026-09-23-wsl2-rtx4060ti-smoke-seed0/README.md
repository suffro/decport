# Open-Jev DecisionCore transfer — bounded smoke test

**Status: smoke test, not evidence.** This run checks that the pipeline works end to end with the
real Open-Jev 2B teacher and real target backbones. It trained on 72 unlabeled decisions for two
epochs and evaluated 36 ID and 36 OOD decisions with one seed. Its transfer metrics are
meaningless at this scale and must not be read as results. No portability claim follows from it.

## What the smoke test established

| Check | Evidence |
|---|---|
| Real Open-Jev 2B loads in the DecPort pipeline | `ZefanCai/Open-Jev-2B@0c7aa49` package verified against manifest `58319da5…`; upstream `jev.model.DecisionModel` loaded `Qwen/Qwen3.5-2B@15852e8` with the rank-8 LoRA (`transfer_smoke/provenance.json`) |
| The frozen core *is* the trained Open-Jev head | Core weights equal the loaded head; re-scoring the hook-captured head input gave a maximum difference of **0.0** on all 480 smoke prompts and on the 116 longest broad-data prompts. In the JevBench teacher run the per-batch 1e-5 abort guard never triggered (that run did not record the value; the script now does) |
| Teacher probabilities are real and non-degenerate | Every smoke train distribution is distinct (24/24 per type); none has max probability above 0.99; mean max probability 0.65 Choice, 0.83 Noul, 0.50 Score; Noul mean P(true) 0.56 (`transfer_smoke/teacher/summary.json`) |
| Gradients flow only through the target adapter | For all 9 trained adapters (3 targets × correct/mismatched/random core): 6/6 adapter tensors received gradients, 0 core tensors, 0 backbone tensors; the core SHA-256 `867bd23f…` was unchanged after every condition |
| Label-free transfer | 0 of 72 transfer inputs and 0 teacher inputs carried answers; the transfer entry point was exercised with a labeled record and rejected it before training |
| All three decision types execute with Open-Jev semantics | Choice, Noul (`[0, s]`, one prompt), and Score ran for the teacher and all targets, with metrics reported per type |
| Controls execute | Untrained adapter, deterministic mismatched teacher (within kind and width), and random frozen core ran for SmolLM2, Gemma 3 270M, and TinyLlama |
| JevBench public integration runs end to end | Teacher and two ported smoke targets ran all 231 public tasks through JevBench's own runner, scoring, and summaries |
| Determinism | This archived run followed the `logits_to_keep=1` memory fix. It reproduced the first smoke run's rendered overall accuracy table exactly for every target, split, and condition. Only that table was compared, because the first run's directory was replaced |

## JevBench public subset (231 of 534 tasks)

Only the public subset is available; these are **not** full-JevBench results. JevBench was pinned at
`2fa63fa` (v1.4.0), whose task loading, scoring, and public data are byte-identical to `f8ce713`,
the commit behind Open-Jev's own published result for this exact 2B release.

### Teacher fidelity

| Measure | Open-Jev 2B via DecPort | Open-Jev published (same release) |
|---|---:|---:|
| Correct / 231 | 151 (65.37%) | 150 (64.94%) |
| Original / Easy / Hard | 56/72, 48/48, 47/111 | 56/72, 48/48, 46/111 |
| Brier mean | 0.47518 | 0.4751 |
| Top-label ECE (10 bins) | 0.1396 | 0.1274 |
| Score ordinal MAE | 0.3510 | 0.3622 |
| Strict-valid vectors | 231/231 | 231/231 |

The same upstream request compiler, candidate batching (one sequence), 16,384-token limit, and
calibrated softmax were used. The result reproduces the published protocol closely but not
bit-exactly: one hard-tier decision differs, and ECE/MAE shift slightly. Five teacher decisions have
top-two probability margins below 0.0062, so a single flip is consistent with bf16 numerical
differences between this runtime (Transformers 5.17.0, PyTorch 2.14, RTX 4060 Ti) and the
published one (Transformers 5.10.2, PyTorch 2.8, H100). Open-Jev does not publish per-item
predictions, so the differing item cannot be identified.

Per type (DecPort run): Choice 89/139, Noul 49/74, Score 13/18.

### Ported smoke targets (pipeline check only)

| System | Correct / 231 | Valid vectors | Refusals |
|---|---:|---:|---|
| Gemma 3 270M + smoke adapter + Open-Jev core | 80 | 231 | none |
| TinyLlama 1.1B + smoke adapter + Open-Jev core | 64 | 194 | 37 × HTTP-422-equivalent context refusals (prompts over TinyLlama's 2,048 positions) |

These adapters saw 72 decisions for two epochs. The numbers only show that ported targets produce
valid JevBench distributions through the frozen Open-Jev core, and that over-context tasks are
refused and scored wrong rather than truncated or silently extrapolated.

## Resources (RTX 4060 Ti 8 GB, WSL2)

| Phase | Time | Peak GPU allocation |
|---|---:|---:|
| Teacher, 480 smoke prompts (68k tokens) incl. load | 29.6 s | 4,221 MiB |
| Teacher worst case, 116 longest broad-data prompts, batch 8 | 3.9 prompts/s, 4.2k tokens/s | 4,364 MiB |
| SmolLM2 / Gemma / TinyLlama caching, 480 prompts incl. load | 5.6 / 6.4 / 11.8 s | 1,693 / 993 / 3,888 MiB |
| Seed 0: 9 adapter trainings + 12 evaluations on cached states | 3.0 s | 40 MiB |
| Whole smoke run (nvidia-smi, incl. CUDA context) | 2 min 14 s | 5,271 MiB used |
| JevBench teacher, 231 tasks, one sequence per call, excluding load | 165 s (p50 0.55 s, p95 2.2 s per task) | not separately recorded |

Before the fix, Gemma 3's 262k-token vocabulary logits made its caching peak at 7,083 MiB on these
short slices and above 10 GB on long prompts. Backbone extraction now requests one logit position;
hidden states were verified bit-identical on all three targets.

## Reproduce

```bash
uv sync --extra dev --extra train --extra openjev
bash scripts/run_openjev_wsl_archive.sh data/jev-broad-v0.1 \
  runs/openjev-transfer-smoke-v0.1-seed0 configs/openjev-transfer-smoke-v0.1.json
uv run python scripts/run_jevbench_public.py --jevbench-root ../jevbench \
  --system openjev-teacher --output runs/jevbench-public-v1.4.0-openjev-2b-teacher
uv run python scripts/run_jevbench_public.py --jevbench-root ../jevbench \
  --system decport-target --backbone-type gemma --model-id unsloth/gemma-3-270m-it \
  --artifact runs/openjev-transfer-smoke-v0.1-seed0/seed-0/targets/gemma/openjev_teacher_distillation \
  --output runs/jevbench-public-v1.4.0-smoke-gemma-openjev-teacher-distillation
```

`../jevbench` must be a clean checkout of `2fa63fa3226cb369795525ed011800f57dcbd894`.

## Archive contents

- `transfer_smoke/`: repository and expanded configs, data manifest, provenance (every pinned
  revision and hash), telemetry, GPU samples, teacher summary and raw smoke logits, per-seed and
  aggregate results, and adapter `config.json` files.
- `jevbench_public/`: JevBench's own public-export aggregates plus per-type and per-tier breakdowns
  and provenance for each run.
- Deliberately excluded: smoke adapter weights (hashes in `excluded_smoke_weights.json`), and
  JevBench per-item records and raw request/response files. Those contain JevBench task content, of
  which only the 72 original tasks are MIT-licensed. Hashes are in `excluded_raw_artifacts.json`.
  The Open-Jev head is not copied; it is identified by package hashes.
- `environment.json` and `SHA256SUMS`.
