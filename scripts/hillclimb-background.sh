#!/usr/bin/env bash
set -euo pipefail

# Run a small, sequential experiment queue after the long Kev-style run.
# Each run is benchmarked on both fixed suites before the next run starts.

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

wait_for_checkpoint() {
  local run_dir="$1"
  while true; do
    if [[ -f "$run_dir/head.pt" ]]; then
      return 0
    fi
    if [[ -f "$run_dir/train.log" ]] && grep -Eq 'Traceback|Error:' "$run_dir/train.log"; then
      echo "training failed: $run_dir" >&2
      tail -n 40 "$run_dir/train.log" >&2
      return 1
    fi
    sleep 60
  done
}

benchmark_run() {
  local run_dir="$1"
  local name="$2"
  wait_for_checkpoint "$run_dir"
  .venv/bin/python -m lev.benchmark --run "$run_dir" \
    --suite benchmarks/banking77-v1 \
    --out "runs/benchmarks/${name}-banking77"
  .venv/bin/python -m lev.benchmark --run "$run_dir" \
    --suite benchmarks/multi-source-v1 \
    --out "runs/benchmarks/${name}-mixed"
}

benchmark_run runs/multi-six-1500 multi-six-1500

LEV_SOURCES=banking77 LEV_N_PER_SOURCE=1500 LEV_EPOCHS=3 LEV_ACCUM=8 \
  ./scripts/train-background.sh runs/banking77-1500-epochs3
benchmark_run runs/banking77-1500-epochs3 banking77-1500-epochs3

LEV_SOURCES=banking77 LEV_N_PER_SOURCE=1500 LEV_EPOCHS=2 LEV_LR=1e-4 LEV_ACCUM=8 \
  ./scripts/train-background.sh runs/banking77-1500-lr1e-4
benchmark_run runs/banking77-1500-lr1e-4 banking77-1500-lr1e-4

date -u +%FT%TZ > runs/hillclimb-complete
