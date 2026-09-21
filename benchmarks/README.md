# Benchmarks

This directory is the versioned record of DecPort experiments. It deliberately separates bounded
pilots from benchmark evidence so exploratory results remain inspectable without overstating the
project's validation status.

## Layout

- [`pilots/`](pilots/) contains bounded runs used to validate the pipeline and decide whether a
  larger experiment is warranted.
- [`diagnostics/`](diagnostics/) contains controlled studies of experiment behavior that are not
  accepted benchmark claims.
- Accepted benchmark results will receive their own top-level collection after a meaningful-scale,
  multi-seed run has been reproduced. None has been accepted yet.

Each recorded run should include a human-readable report, the runner's `results.json`, environment
and data provenance, checksums, lightweight safetensors artifacts, and relevant resource telemetry.
Base-model checkpoints and downloaded upstream datasets do not belong in Git.

`scripts/run_experiment.py` still writes working output to ignored `runs/`. A run is copied here
only when it is intentionally retained as project evidence.
