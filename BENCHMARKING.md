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

Reports include headline accuracy, NLL, calibration, confidence, latency, and
for mixed suites a `by_source` section with separate accuracy/NLL for each
dataset. The headline number answers “did the mixture improve overall?” while
the source breakdown answers “which task improved or regressed?”

The companion [`multi-source-v1`](benchmarks/multi-source-v1) suite contains
900 fixed held-out records: 150 from each of Kev's six sources. It uses one
classification question per record and permits different option counts across
sources. Use it to measure generalization across the mixed training recipe;
use `banking77-v1` to catch regressions on the original task.

```bash
uv run python -m lev.benchmark \
  --run runs/candidates/multi-six-1500 \
  --suite benchmarks/multi-source-v1 \
  --out runs/benchmarks/multi-six-1500-mixed
```

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
- `multi-source-v1`: the current six-source mixture: Banking77, BoolQ, AG News,
  MNLI, SST-5, and Yelp.
- `finance-v1`: a future finance-focused suite should be separate from the
  general mixture; Financial PhraseBank is sentiment, not transaction-intent
  classification, so it should not silently redefine the Banking77 target.

Never rewrite `banking77-v1`. Its purpose is to make progress and regressions
visible over time.
