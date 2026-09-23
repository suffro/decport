# Open-Jev 2B DecisionCore transfer — five seeds, three targets

**Status: accepted benchmark evidence, reproduced bit-exactly.** The acceptance certifies the
measurement, including its negative findings. It is **not** a general portability claim.

## Outcome

The evidence **partially supports** DecPort.

- **Transferred.** Input-specific Open-Jev decision behavior transfers without target labels to all
  three frozen targets. The correct teacher beats the mismatched-teacher control overall on ID and
  OOD in 5/5 seeds for every target. The effect comes from Choice and Score.
- **Not shown: the pretrained core matters.** A random frozen core with the same norm, bias, and
  temperature, trained on the same teacher, matches the real Open-Jev core within noise on every
  target. The adapter does essentially all of the work. Decision 0006 anticipated this for a
  rank-one linear core.
- **Noul does not work.** SmolLM2 and TinyLlama learn no input-specific Noul behavior, even on
  their training decisions, and collapse to "false". Gemma learns a little. This is an unresolved
  limitation.
- **Weaker OOD.** Gains are much smaller OOD than ID, and Gemma Choice OOD shows no gain.
- **JevBench public subset.** The ported targets score 78–92 of 231, against 151 for the teacher
  and about 73 expected from uniform guessing.

## Protocol (unchanged from `configs/openjev-transfer-v0.1.json`)

- **Teacher.** Frozen `ZefanCai/Open-Jev-2B@0c7aa49` on `Qwen/Qwen3.5-2B@15852e8`, run once. Its
  trained `Linear(2048→1)` head and calibration T = 1.5188 form one frozen DecisionCore
  (SHA-256 `867bd23f…`), shared by every target, seed, and condition.
- **Targets.** Frozen SmolLM2-360M-Instruct, Gemma 3 270M Instruct, and TinyLlama-1.1B-Chat. Only a
  `d → 256 → 2048` DecPort adapter trains.
- **Data.** `data/jev-broad-v0.1`: 4,500 train decisions (1,500 per type), 2,000 ID
  (570 Choice / 700 Noul / 730 Score), and 1,500 OOD (500 per type). ID is ARC-Easy / BoolQ /
  Yelp; OOD is OpenBookQA / QNLI / Amazon. Labels are used only for evaluation.
- **Training.** Seeds 0–4, 10 epochs, AdamW lr 1e-3, batch 8. The objective is
  KL(teacher ‖ student) at T = 1 plus 0.1 × centered-logit MSE on calibrated typed logits.
- **Conditions per target.** All four conditions start from one adapter state:
  - untrained adapter;
  - correct Open-Jev teacher;
  - mismatched teacher (derangement within decision kind and option count);
  - random frozen core trained on the correct teacher.
- Values are mean ± sample SD over five seeds.

### Deviation: one operational fix

The first attempt aborted at seed 0, before any adapter trained (`attempt1_failure.json`). The
train split has exactly one 5-option Choice decision, and a derangement within its kind and
option count is impossible for a singleton group. Commit `0ab7155` lets that decision keep its own
teacher output. It is recorded as a fixed point: 1 of 4,500 decisions per seed, verified by the
audit. Every other output is still deranged within kind and width. The fixed point can only make
the mismatched control *closer* to the correct teacher, so it works against the hypothesis. The
teacher pass of the failed attempt was byte-identical to the accepted run.

## A. Correct teacher vs mismatched teacher (overall)

| Target | Split | Teacher | Correct | Mismatch | Untrained | Random core | Δ vs mismatch | Seeds > mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2 | ID | 0.7310 | 0.4602 ± 0.0074 | 0.2823 ± 0.0281 | 0.3176 ± 0.0449 | 0.4588 ± 0.0082 | +0.1779 ± 0.0266 | 5/5 |
| SmolLM2 | OOD | 0.6467 | 0.3755 ± 0.0093 | 0.3112 ± 0.0200 | 0.3219 ± 0.0119 | 0.3716 ± 0.0047 | +0.0643 ± 0.0206 | 5/5 |
| Gemma 3 | ID | 0.7310 | 0.4659 ± 0.0284 | 0.2806 ± 0.0249 | 0.3202 ± 0.0396 | 0.4370 ± 0.0119 | +0.1853 ± 0.0284 | 5/5 |
| Gemma 3 | OOD | 0.6467 | 0.3799 ± 0.0135 | 0.3115 ± 0.0160 | 0.3268 ± 0.0105 | 0.3672 ± 0.0091 | +0.0684 ± 0.0086 | 5/5 |
| TinyLlama | ID | 0.7310 | 0.4478 ± 0.0023 | 0.2727 ± 0.0140 | 0.3190 ± 0.0311 | 0.4453 ± 0.0029 | +0.1751 ± 0.0141 | 5/5 |
| TinyLlama | OOD | 0.6467 | 0.3929 ± 0.0048 | 0.3208 ± 0.0049 | 0.3044 ± 0.0095 | 0.3875 ± 0.0085 | +0.0721 ± 0.0024 | 5/5 |

Overall teacher top-choice agreement for the correct condition is 0.52–0.53 ID and 0.43–0.48 OOD.
Ported targets recover roughly 61–64% of the teacher's ID accuracy and 58–61% of its OOD accuracy.

## B. Pretrained Open-Jev core vs random frozen core

| Target | ID Δ acc. | OOD Δ acc. | Seeds > random (ID / OOD) | ID KL: Open-Jev / random |
|---|---:|---:|---:|---:|
| SmolLM2 | +0.0014 ± 0.0112 | +0.0039 ± 0.0054 | 3/5 / 4/5 | 0.2805 / 0.2877 |
| Gemma 3 | +0.0289 ± 0.0319 | +0.0127 ± 0.0202 | 4/5 / 3/5 | 0.3003 / 0.3234 |
| TinyLlama | +0.0025 ± 0.0033 | +0.0055 ± 0.0060 | 4/5 / 5/5 | 0.3082 / 0.3088 |

No difference is distinguishable from seed noise. Gemma's +0.029 is the largest, and it is within
one SD. Per-type teacher agreement, KL, and correlation are also nearly identical between the two
cores. Because the core is a rank-one linear readout, the adapter's last layer can realize any
readout direction. A random direction with matched scale is therefore essentially as good as the
pretrained one. **This experiment gives no evidence that the specific pretrained Open-Jev core
contributes reusable structure.**

## C. Choice, Noul, and Score separately

Accuracy, mean ± SD over five seeds. Δ columns are correct-teacher gains.

| Target | Split | Type | Teacher | Correct | Mismatch | Untrained | Random core | Δ vs mismatch | Δ vs random |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2 | ID | Choice | 0.8877 | 0.6446 ± 0.0118 | 0.2253 ± 0.0588 | 0.2586 ± 0.0404 | 0.6414 ± 0.0149 | +0.4193 | +0.0032 |
| SmolLM2 | ID | **Noul** | 0.7929 | 0.3631 ± 0.0056 | 0.3849 ± 0.0532 | 0.4726 ± 0.1515 | 0.3680 ± 0.0140 | −0.0217 | −0.0049 |
| SmolLM2 | ID | Score | 0.5493 | 0.4093 ± 0.0145 | 0.2285 ± 0.0121 | 0.2151 ± 0.0229 | 0.4033 ± 0.0087 | +0.1808 | +0.0060 |
| SmolLM2 | OOD | Choice | 0.6080 | 0.3136 ± 0.0111 | 0.2296 ± 0.0271 | 0.2684 ± 0.0415 | 0.3080 ± 0.0097 | +0.0840 | +0.0056 |
| SmolLM2 | OOD | **Noul** | 0.8600 | 0.5176 ± 0.0132 | 0.5096 ± 0.0311 | 0.5048 ± 0.0263 | 0.5240 ± 0.0000 | +0.0080 | −0.0064 |
| SmolLM2 | OOD | Score | 0.4720 | 0.2952 ± 0.0186 | 0.1944 ± 0.0065 | 0.1924 ± 0.0156 | 0.2828 ± 0.0084 | +0.1008 | +0.0124 |
| Gemma 3 | ID | Choice | 0.8877 | 0.4596 ± 0.0135 | 0.2330 ± 0.0426 | 0.2589 ± 0.0344 | 0.4389 ± 0.0170 | +0.2267 | +0.0207 |
| Gemma 3 | ID | **Noul** | 0.7929 | 0.5066 ± 0.0669 | 0.3954 ± 0.0347 | 0.4863 ± 0.1219 | 0.5020 ± 0.0199 | +0.1111 | +0.0046 |
| Gemma 3 | ID | Score | 0.5493 | 0.4318 ± 0.0176 | 0.2077 ± 0.0206 | 0.2088 ± 0.0162 | 0.3732 ± 0.0205 | +0.2241 | +0.0586 |
| Gemma 3 | OOD | Choice | 0.6080 | 0.2580 ± 0.0147 | 0.2608 ± 0.0296 | 0.2740 ± 0.0359 | 0.2568 ± 0.0044 | −0.0028 | +0.0012 |
| Gemma 3 | OOD | **Noul** | 0.8600 | 0.5340 ± 0.0224 | 0.4796 ± 0.0387 | 0.5024 ± 0.0288 | 0.5260 ± 0.0032 | +0.0544 | +0.0080 |
| Gemma 3 | OOD | Score | 0.4720 | 0.3476 ± 0.0293 | 0.1940 ± 0.0079 | 0.2040 ± 0.0088 | 0.3188 ± 0.0287 | +0.1536 | +0.0288 |
| TinyLlama | ID | Choice | 0.8877 | 0.4881 ± 0.0052 | 0.2463 ± 0.0313 | 0.2284 ± 0.0158 | 0.4807 ± 0.0135 | +0.2418 | +0.0074 |
| TinyLlama | ID | **Noul** | 0.7929 | 0.3606 ± 0.0022 | 0.3617 ± 0.0012 | 0.5340 ± 0.1099 | 0.3614 ± 0.0000 | −0.0011 | −0.0009 |
| TinyLlama | ID | Score | 0.5493 | 0.5000 ± 0.0078 | 0.2079 ± 0.0156 | 0.1836 ± 0.0296 | 0.4981 ± 0.0076 | +0.2921 | +0.0019 |
| TinyLlama | OOD | Choice | 0.6080 | 0.3028 ± 0.0095 | 0.2500 ± 0.0206 | 0.2400 ± 0.0123 | 0.3020 ± 0.0117 | +0.0528 | +0.0008 |
| TinyLlama | OOD | **Noul** | 0.8600 | 0.5052 ± 0.0076 | 0.5236 ± 0.0009 | 0.4916 ± 0.0205 | 0.4948 ± 0.0198 | −0.0184 | +0.0104 |
| TinyLlama | OOD | Score | 0.4720 | 0.3708 ± 0.0064 | 0.1888 ± 0.0098 | 0.1816 ± 0.0131 | 0.3656 ± 0.0129 | +0.1820 | +0.0052 |

- **Choice.** Strong ID transfer on every target (+0.23 to +0.42 over mismatch, 5/5 seeds). OOD is
  positive for SmolLM2 and TinyLlama (5/5 seeds) and absent for Gemma (3/5, mean −0.003).
- **Score.** The most robust type. It beats the mismatched teacher ID and OOD on every target in
  5/5 seeds. Ordinal MAE falls from 1.87–2.06 (mismatch) to 0.70–0.97 ID and 1.01–1.34 OOD;
  the teacher's is 0.574 / 0.752.
- **Noul: not working.**
  - SmolLM2 and TinyLlama predict "true" for about 1% of ID decisions. Their ID accuracy of 0.36
    equals the false-label rate (1 − 0.6386), below the untrained adapter.
  - On their own **training** decisions, their Noul teacher correlation is 0.063 ± 0.010 and
    0.048 ± 0.024. Their Noul KL barely moves from the untrained adapter (0.2885→0.2858 and
    0.2869→0.2867), while Choice and Score KL fall to roughly half.
  - The students reproduce only the teacher's marginal P(true) of about 0.48, which is
    underfitting of input-specific behavior, not an OOD effect.
  - Gemma learns some Noul behavior (train correlation 0.300 ± 0.104; ID +0.11 over mismatch, 5/5
    seeds). It is still far below the teacher (0.51 vs 0.79 ID) and no better than its random core.
  - One TinyLlama seed (seed 0) diverges on QNLI OOD (P(true) 0.92, Noul KL 2.43). This produces
    the large TinyLlama OOD Noul KL SD.

## ID vs OOD

ID gains over mismatch are +0.175 to +0.185; OOD gains shrink to +0.064 to +0.072. Only Score
transfers robustly under dataset shift. Choice OOD holds for two of three targets. Noul is at or near
chance OOD for every target (the QNLI majority-class rate is 0.524), while the teacher reaches
0.86.

## Calibration

Overall ID NLL / Brier improve over the untrained and mismatched adapters on every target. ID ECE
worsens (0.066–0.116 vs 0.047–0.060 untrained), because the untrained adapter is near-uniform and
therefore trivially calibrated. OOD NLL, Brier, and ECE are worse than the untrained adapter on
every target. TinyLlama OOD is the worst (NLL 1.44 ± 0.31). The teacher's reference is
ECE 0.081 ID and 0.075 OOD. The full NLL / Brier / ECE / KL tables are in
`transfer/report_tables.md`.

## JevBench v1.4.0 — public subset (231 of 534 tasks)

**Only the 231 public tasks are available. These are not full-JevBench results.**

The adapters were chosen before evaluation by a label-free rule: for each target, the
correct-teacher adapter with the lowest final-epoch training loss. That selected SmolLM2 seed 0,
Gemma seed 0, and TinyLlama seed 1. No JevBench answers or evaluation labels were used for
selection or tuning.

| System | Correct / 231 | Choice / 139 | Noul / 74 | Score / 18 | Original / Easy / Hard | Brier | ECE | Top-choice agreement with teacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Open-Jev 2B teacher | **151** (65.4%) | 89 | 49 | 13 | 56 / 48 / 47 | 0.4752 | 0.140 | — |
| DecPort + SmolLM2 | 92 (39.8%) | 51 | 37 | 4 | 28 / 21 / 43 | 0.7474 | 0.166 | 79 |
| DecPort + Gemma 3 | 81 (35.1%) | 41 | 37 | 3 | 24 / 23 / 34 | 0.7042 | 0.125 | 100 |
| DecPort + TinyLlama | 78 (33.8%) | 38 | 36 | 4 | 28 / 27 / 23 | 0.8556 | 0.309 | 88 |
| Uniform guess (expected) | 73.4 (31.8%) | | | | | | | |

- The teacher again reproduced 151/231 (Brier 0.47518), identical to the smoke fidelity check.
  Open-Jev published 150/231 for this release.
- TinyLlama refused 37 tasks whose prompts exceed its 2,048-token context. They were scored wrong,
  not truncated; it is 78/194 on the tasks it could run.
- Ported Noul is at chance (36–37/74), matching the Noul failure above.
- The ported targets are far below the teacher and modestly above uniform guessing. The Gemma
  adapter (81) scores about the same as its 72-decision smoke adapter (80).

## Reproducibility

- The reproduction (`reproduction/`) re-ran the identical command, commit, data, and hardware. Its
  audit also passed. Details are in `reproduction/reproduction_check.json`.
- All 70 result JSON files match with maximum numeric difference 0.0. All 75 safetensors artifacts,
  the teacher logits, and `aggregate_summary.json` are byte-identical. Its adapter weights are
  therefore not duplicated here.
- `scripts/audit_openjev_results.py` passed on both runs (`audit.json`, `reproduction/audit.json`).
  It verifies:
  - teacher/core parity of 0.0;
  - the same core digest before, after, and when independently reloaded from the pinned package;
  - that the untrained adapter did not train;
  - that gradients reached all adapter tensors and no core or backbone tensor, in every trained
    condition;
  - zero answers in teacher and distillation inputs, and that the label guard fired;
  - that every seed, target, and condition is present;
  - that per-type and per-dataset counts equal the data (no decision is dropped from any metric);
  - that the mismatched-control fixed point is exactly the one expected;
  - that all numbers are finite.
- Its failure path was exercised on a tampered copy: an injected core gradient and a dropped Noul
  decision were both rejected.
- Wall time was 50 min per run: teacher 13 min for 26,627 candidate prompts and 3.84M tokens; about
  5 min per seed. Peak allocation was 4,398 MiB for the teacher, 5,105 MiB for TinyLlama caching,
  and 40 MiB for training (`transfer/telemetry.json`, `transfer/gpu_usage.csv`).

```bash
uv sync --extra dev --extra train --extra openjev
bash scripts/run_openjev_wsl_archive.sh data/jev-broad-v0.1 runs/openjev-transfer-v0.1-seeds0-4
uv run python scripts/audit_openjev_results.py --run runs/openjev-transfer-v0.1-seeds0-4 \
  --data data/jev-broad-v0.1 --verify-core
uv run python scripts/render_openjev_report.py --run runs/openjev-transfer-v0.1-seeds0-4
uv run python scripts/run_jevbench_public.py --jevbench-root ../jevbench --system decport-target \
  --backbone-type smollm --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --artifact runs/openjev-transfer-v0.1-seeds0-4/seed-0/targets/smollm/openjev_teacher_distillation \
  --output runs/jevbench-public-v1.4.0-transfer-v0.1-smollm-seed0-openjev-teacher-distillation
```

`../jevbench` must be a clean checkout of `2fa63fa3226cb369795525ed011800f57dcbd894`.

## Archive contents

- `transfer/` is the complete run:
  - repository and expanded configs, data manifest, and provenance (every pinned revision and
    hash);
  - telemetry, GPU samples, and start/end times;
  - teacher summary and raw logits;
  - per-seed `results.json`, `aggregate_summary.json`, and rendered `report_tables.md`;
  - all 75 adapter / control-core safetensors artifacts, with the Open-Jev head excluded because it
    is identified by package hashes.
- `reproduction/` holds the reproduction's aggregate, provenance, telemetry, GPU samples, audit,
  and the comparison.
- `jevbench_public/` holds JevBench's own reports plus provenance for each system, `summary.json`
  (uniform baseline and teacher agreement), and hashes of the excluded per-item records, ledgers,
  and raw files (`excluded_raw_artifacts.json`). Those contain JevBench task content and stay out
  of Git.
- `attempt1_failure.json`, `audit.json`, `environment.json`, and `SHA256SUMS`.
