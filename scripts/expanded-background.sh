#!/usr/bin/env bash
set -euo pipefail

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

sources="banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,imdb,amazon,arc,openbookqa,csqa"

# Wait until the current six-source queue is done so MPS training is sequential.
wait_for_checkpoint runs/multi-six-1500-epochs3

if [[ ! -d benchmarks/multi-source-v2 ]]; then
  .venv/bin/python scripts/make_benchmark.py \
    --sources "$sources" \
    --n_per_source 150 \
    --seed 7 \
    --name multi-source-v2 \
    --out benchmarks/multi-source-v2
fi

LEV_SOURCES="$sources" LEV_N_PER_SOURCE=750 LEV_EPOCHS=2 LEV_ACCUM=8 \
  ./scripts/train-background.sh runs/kev-thirteen-750
wait_for_checkpoint runs/kev-thirteen-750

.venv/bin/python -m lev.benchmark \
  --run runs/kev-thirteen-750 \
  --suite benchmarks/banking77-v1 \
  --out runs/benchmarks/kev-thirteen-750-banking77
.venv/bin/python -m lev.benchmark \
  --run runs/kev-thirteen-750 \
  --suite benchmarks/multi-source-v1 \
  --out runs/benchmarks/kev-thirteen-750-six-source
.venv/bin/python -m lev.benchmark \
  --run runs/kev-thirteen-750 \
  --suite benchmarks/multi-source-v2 \
  --out runs/benchmarks/kev-thirteen-750-expanded

date -u +%FT%TZ > runs/expanded-complete
