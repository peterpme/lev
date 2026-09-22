#!/usr/bin/env bash
set -euo pipefail

public_run="runs/phase1-public-10"
synthetic_run="runs/phase2-public-synthetic-12"
log="${synthetic_run}/train.log"

mkdir -p "$synthetic_run"
while [[ ! -f "${public_run}/head.pt" ]]; do
  sleep 60
done

exec env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m lev.train \
  --sources banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,imdb,amazon,legacy_policy,compositional \
  --n_per_source 1000 \
  --epochs 2 \
  --lr 0.0002 \
  --lora 16 \
  --accum 8 \
  --out "$synthetic_run" \
  > "$log" 2>&1
