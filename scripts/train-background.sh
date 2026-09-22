#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-runs/banking77-smoke}"
n_per_source="${LEV_N_PER_SOURCE:-40}"
sources="${LEV_SOURCES:-banking77}"
epochs="${LEV_EPOCHS:-1}"
accum="${LEV_ACCUM:-4}"
lr="${LEV_LR:-2e-4}"
lora="${LEV_LORA:-16}"
run_name=$(basename "${run_dir%/}" | tr -cd '[:alnum:]_-')
session_name="${LEV_SESSION_NAME:-lev-${run_name}}"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

mkdir -p "$run_dir"
: > "$run_dir/train.log"

command="exec /usr/bin/env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false uv run python -m lev.train --sources '$sources' --n_per_source '$n_per_source' --epochs '$epochs' --lr '$lr' --lora '$lora' --accum '$accum' --out '$run_dir' > '$run_dir/train.log' 2>&1"
tmux new-session -d -s "$session_name" -c "$PWD" "$command"

pid=$(tmux display-message -p -t "$session_name:0.0" '#{pane_pid}')
echo "$pid" > "$run_dir/train.pid"
echo "$session_name" > "$run_dir/train.session"
echo "started session=$session_name sources=$sources pid=$pid log=$run_dir/train.log"
