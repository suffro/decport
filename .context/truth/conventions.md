# Conventions

## Repository conventions

- Python packages use the `src/` layout and support Python 3.10+.
- Public decision data uses `state`, `question`, `options`, and optional `answer`.
- Runtime artifacts use `config.json` plus safetensors component files.
- Dependencies stay limited to PyTorch, Transformers, safetensors, the optional datasets extra, and
  the optional `openjev` extra (pinned Open-Jev, PEFT, Accelerate) for the external teacher.
- External systems (Open-Jev, JevBench) are used, not forked: pin exact revisions, verify hashes,
  and import their own code instead of copying it.

## Development workflow

- Install with `uv sync --extra dev --extra train --extra openjev`.
- Run `uv run ruff check .` and `uv run pytest` after meaningful changes.
- Real checkpoint tests require `DECPORT_RUN_REAL_MODELS=1` and are not part of ordinary CI.
  `DECPORT_JEVBENCH_ROOT` enables the pinned JevBench checkout test.
- In the Windows + WSL2 setup, run Git from Windows; WSL Git sees every file as modified because of
  DrvFs mode bits. Run Python, uv, and GPU jobs in WSL.
- Run `syngraphe check` before completing substantial work.

## Important rules

- Never unfreeze a language-model backbone in the v0.1 experiment.
- Never retrain, modify, or replace the Open-Jev teacher or its decision core; only target adapters
  learn, and never from target labels.
- Do not report tiny smoke runs as benchmark results.
- Never commit JevBench task content, per-item records, or raw responses, and never tune against
  JevBench answers. Label JevBench numbers as public-subset results.
- Keep the head independent of a backbone's native hidden width.
- Do not add postponed serving, UI, kernel, quantization, or deployment scope.
