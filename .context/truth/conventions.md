# Conventions

## Repository conventions

- Python packages use the `src/` layout and support Python 3.10+.
- Public decision data uses `state`, `question`, `options`, and optional `answer`.
- Runtime artifacts use `config.json` plus safetensors component files.
- Dependencies stay limited to PyTorch, Transformers, safetensors, and the optional datasets extra.

## Development workflow

- Install with `uv sync --extra dev --extra train`.
- Run `uv run ruff check .` and `uv run pytest` after meaningful changes.
- Real checkpoint tests require `DECPORT_RUN_REAL_MODELS=1` and are not part of ordinary CI.
- Run `syngraphe check` before completing substantial work.

## Important rules

- Never unfreeze a language-model backbone in the v0.1 experiment.
- Do not report tiny smoke runs as benchmark results.
- Keep the head independent of a backbone's native hidden width.
- Do not add postponed serving, UI, kernel, quantization, or deployment scope.
