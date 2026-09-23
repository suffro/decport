#!/usr/bin/env bash
# Run the Open-Jev DecisionCore transfer experiment with GPU telemetry and archive inputs.
set -uo pipefail

data_dir="${1:?data directory is required}"
output_dir="${2:?output directory is required}"
config="${3:-configs/openjev-transfer-v0.1.json}"
mkdir -p "$output_dir"
cp "$data_dir/data_manifest.json" "$output_dir/data_manifest.json"
cp "$config" "$output_dir/repository_config.json"
date --iso-8601=seconds > "$output_dir/run_start.txt"
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 1 > "$output_dir/gpu_usage.csv" &
monitor_pid=$!
uv run python scripts/run_openjev_experiment.py \
  --config "$config" \
  --train "$data_dir/train.jsonl" \
  --eval "$data_dir/eval.jsonl" \
  --ood "$data_dir/ood.jsonl" \
  --output "$output_dir"
status=$?
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
date --iso-8601=seconds > "$output_dir/run_end.txt"
exit "$status"
