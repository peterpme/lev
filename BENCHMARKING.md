# Benchmarking Lev

## What is saved today?

The first stable suite is [`benchmarks/banking77-v1`](benchmarks/banking77-v1):

- `test.jsonl` contains 150 fixed Banking77 test records;
- `manifest.json` records the suite version and SHA-256 checksum;
- `baselines/lev-banking77-1500.json` records the first larger-model result.

The trained weights remain in `runs/`, which is intentionally ignored by Git.
The committed baseline is the durable summary; the checkpoint itself must be
retrained or stored separately if someone needs to reproduce the exact model.

## Run a candidate

Give every candidate its own output directory:

```bash
uv run python -m lev.train \
  --n_per_source 1500 \
  --epochs 2 \
  --accum 8 \
  --out runs/candidates/lr-2e-4

uv run python -m lev.benchmark \
  --run runs/candidates/lr-2e-4 \
  --suite benchmarks/banking77-v1 \
  --out runs/benchmarks/lr-2e-4
```

The benchmark refuses to run if the fixed test file has changed. Keep each
candidate report so experiments can be compared later.

## How to hill-climb safely

Change one meaningful variable at a time:

```text
training examples
learning rate
number of epochs
LoRA rank
augmentation probabilities
base model
```

Use a development set for repeated experimentation. Treat `banking77-v1` as a
locked test set: run it after choosing a candidate, not after every tiny tweak.
Otherwise the model-selection process gradually overfits the benchmark.

For each candidate, prefer:

1. lower NLL;
2. higher accuracy;
3. lower ECE;
4. no severe regression on confidence-at-90% accuracy;
5. inspection of a few wrong and overconfident examples.

Do not optimize for confidence by itself. A model that assigns 0.99 to wrong
answers is worse calibrated even if it looks decisive.

## Future suites

Create a new benchmark version when the evaluation question changes:

- `banking77-v2`: different presentation or additional Banking77 cases;
- `transfer-v1`: datasets Lev never trained on;
- `multi-source-v1`: BoolQ, AG News, MNLI, SST-5, and Yelp after they are added.

Never rewrite `banking77-v1`. Its purpose is to make progress and regressions
visible over time.
