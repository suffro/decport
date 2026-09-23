# Benchmarks

This directory is the versioned record of DecPort experiments. It deliberately separates bounded
pilots from benchmark evidence so exploratory results remain inspectable without overstating the
project's validation status.

## Layout

- [`pilots/`](pilots/) contains bounded runs used to validate the pipeline and decide whether a
  larger experiment is warranted.
- [`diagnostics/`](diagnostics/) contains controlled studies of experiment behavior that are not
  accepted benchmark claims.
- [`accepted/`](accepted/) contains meaningful-scale, multi-seed runs that passed their audit and
  were reproduced before acceptance. Acceptance certifies the measurement, including negative
  findings, not a claim beyond the run's report.

Each recorded run should include a human-readable report, the runner's `results.json`, environment
and data provenance, checksums, lightweight safetensors artifacts, and relevant resource telemetry.
Base-model checkpoints and downloaded upstream datasets do not belong in Git. External-system
artifacts follow their own terms: Open-Jev weights are identified by package hashes rather than
copied, and JevBench per-item records or raw responses stay out of Git; only its public-export
aggregates are archived, always labeled as public-subset results.

`scripts/run_experiment.py` still writes working output to ignored `runs/`. A run is copied here
only when it is intentionally retained as project evidence.
