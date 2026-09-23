#!/usr/bin/env bash
# Run the DecPort final ship gate with GPU telemetry and archive its inputs.
set -uo pipefail

data_dir="${1:?transfer data directory is required}"
source_dir="${2:?source-core data directory is required}"
output_dir="${3:?output directory is required}"
config="${4:-configs/decport-final-gate-v0.1.json}"
mkdir -p "$output_dir"
cp "$data_dir/data_manifest.json" "$output_dir/data_manifest.json"
cp "$source_dir/data_manifest.json" "$output_dir/source_core_data_manifest.json"
cp "$config" "$output_dir/repository_config.json"
sha256sum "$config" > "$output_dir/repository_config.sha256"
date --iso-8601=seconds > "$output_dir/run_start.txt"
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 5 > "$output_dir/gpu_usage.csv" &
monitor_pid=$!
uv run python scripts/run_final_gate.py \
  --config "$config" \
  --data "$data_dir" \
  --source-data "$source_dir" \
  --output "$output_dir"
status=$?
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
date --iso-8601=seconds > "$output_dir/run_end.txt"
exit "$status"
