# Lev

Lev is a small, educational reimplementation of Kev's original Qwen2.5-0.5B decision model.

The first milestone is deliberately narrow:

```text
Banking77 message + 77 intents -> probability for every intent
```

Lev uses the same core mechanism as Kev's first commit: a Qwen2.5-0.5B backbone, rank-16 LoRA, a block-causal question mask, and a learned pointer head. It does not generate text.

## Setup

Requires Python 3.12 or 3.13 and `uv`.

```bash
uv sync --python 3.13
```

The public dataset and model are loaded from the Hugging Face Hub. This machine has a stale implicit Hub login, so commands below disable implicit authentication while accessing public assets.

## Inspect Banking77

Load and display one record directly from Hugging Face:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python -m lev.render
```

Optionally save a small inspection set:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python -m lev.data \
  --n_per_source 40 \
  --out data/banking77-smoke.jsonl
```

Training does not require JSONL; it loads and transforms Banking77 in memory, as Kev does.

## Verify the model path

This downloads Qwen2.5-0.5B, builds the LoRA adapter and pointer head, and performs one forward pass without training:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python -m lev.train \
  --n_per_source 1 \
  --dry_run
```

## Start a background smoke run

```bash
./scripts/train-background.sh
```

The default run trains on 40 Banking77 records for one epoch and writes:

```text
runs/banking77-smoke/train.log
runs/banking77-smoke/train.pid
runs/banking77-smoke/adapter_model.safetensors
runs/banking77-smoke/head.pt
```

Watch it with:

```bash
tail -f runs/banking77-smoke/train.log
```

Reload the saved LoRA adapter and pointer head, then evaluate held-out rows:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python -m lev.evaluate \
  --run runs/banking77-smoke \
  --n 40
```

This writes `runs/banking77-smoke/eval.json`. A 40-row smoke run only proves the
pipeline works; it is far too small to expect useful accuracy.

For a larger run:

```bash
LEV_N_PER_SOURCE=1500 LEV_EPOCHS=2 LEV_ACCUM=8 \
  ./scripts/train-background.sh runs/banking77-1500
```

That matches Kev's released example count per source, but trains Banking77 only.
Check whether its detached session is active with:

```bash
tmux has-session -t lev-banking77-1500 && echo running || echo finished
```

## Current boundary

Banking77 Choice training comes first. TypeSafe API compatibility, `noul`, `score`, six-dataset training, calibration, and serving come later.
