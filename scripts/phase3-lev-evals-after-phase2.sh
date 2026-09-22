#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f runs/phase1-public-10/head.pt || ! -f runs/phase2-public-synthetic-12/head.pt ]]; do
  sleep 60
done

for run_name in phase1-public-10 phase2-public-synthetic-12; do
  for suite_name in decision-v7 transfer-v4 transfer-v9; do
    out="runs/benchmarks/${run_name}-${suite_name}"
    if [[ -f "${out}/report.json" ]]; then
      continue
    fi
    mkdir -p "runs/benchmarks"
    .venv/bin/python -m lev.kev_benchmark \
      --run "runs/${run_name}" \
      --suite "benchmarks/kev/${suite_name}" \
      --split development \
      --out "$out"
  done
done
