# Lev

Lev is a small, educational decision model inspired by [Kev](https://github.com/jaredpalmer/kev).
It adapts the small [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)
language model to answer structured multiple-choice questions.

The first version intentionally uses one dataset:

```text
Banking77 message + 77 banking intents → one probability per intent
```

Lev does not generate an answer as prose. It scores the choices supplied by the
caller and returns JSON.

## TL;DR

- Qwen 0.5B provides general language understanding.
- LoRA makes a small, efficient adaptation to Lev's decision format.
- A pointer head produces one score for each supplied option.
- Softmax turns those scores into probabilities.
- FastAPI returns the selected option and its probabilities.

This is a learning project and a working prototype, not a production decision
system.

## Quick start

You need Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/peterpme/lev.git
cd lev
uv sync --extra serve
```

The first run downloads Qwen and Banking77 from Hugging Face.

### 1. Train a small model

This smoke run is intentionally small and is mainly useful for verifying the
pipeline:

```bash
uv run python -m lev.train \
  --n_per_source 40 \
  --epochs 1 \
  --accum 4 \
  --out runs/banking77-smoke
```

Evaluate the saved checkpoint on held-out Banking77 examples:

```bash
uv run python -m lev.evaluate \
  --run runs/banking77-smoke \
  --n 40
```

For a more useful first model, train 1,500 examples for two epochs:

```bash
uv run python -m lev.train \
  --n_per_source 1500 \
  --epochs 2 \
  --accum 8 \
  --out runs/banking77-1500
```

The folder name is only a label: it identifies which training run the server
should load.

To mirror Kev's six-source recipe, pass the source list explicitly:

```bash
LEV_SOURCES=banking77,boolq,agnews,mnli,sst5,yelp \
  ./scripts/train-background.sh runs/multi-six-40
```

This trains 40 records from each source for one epoch by default. The loaders
and internal question types are implemented; the public API currently exposes
the same three types, with `choice` being the simplest path to try first.

Kev's later recipe adds more optional sources. Lev also includes compatible
loaders for `trec`, `dbpedia14`, `imdb`, `amazon`, `arc`, `openbookqa`, and
`csqa`. Add them deliberately to an experiment rather than assuming more data
is automatically better: these sources teach different tasks and can improve
transfer while slightly diluting the Banking77 objective.

```bash
LEV_SOURCES=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,imdb,amazon,arc,openbookqa,csqa \
  LEV_N_PER_SOURCE=750 LEV_EPOCHS=2 \
  ./scripts/train-background.sh runs/kev-thirteen-750
```

For a repeatable expanded evaluation, first create a new frozen suite from the
same held-out splits, then benchmark against it:

```bash
uv run python scripts/make_benchmark.py \
  --sources banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,imdb,amazon,arc,openbookqa,csqa \
  --n_per_source 150 --name multi-source-v2 \
  --out benchmarks/multi-source-v2
```

For a finance-specific experiment, Financial PhraseBank is also available as
an optional source:

```bash
LEV_SOURCES=banking77,financial_phrasebank \
  ./scripts/train-background.sh runs/banking-finance-smoke
```

It contributes three-way financial sentiment examples (`negative`, `neutral`,
and `positive`). It remains optional until the six-source benchmark is measured.

Financial PhraseBank is useful for testing financial language and sentiment,
but it is not a Backpack transaction-intent dataset. The most valuable future
Backpack-specific data would be anonymized, labeled support messages or
decision examples from the product domain, kept in a separate benchmark.

### 2. Start the API

```bash
uv run python -m lev.serve \
  --run runs/banking77-smoke \
  --port 8008
```

In another terminal:

```bash
curl -s http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lev-latest",
    "state": "I need to exchange currencies using my mobile banking app.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "Which banking intent best describes this request?",
        "criteria": {
          "activate_my_card": "Activating a card",
          "exchange_via_app": "Exchanging currencies in the app",
          "cash_withdrawal": "Taking out cash"
        }
      }
    }
  }'
```

The response contains the selected option, a confidence summary, and the full
probability distribution. Replace `runs/banking77-smoke` with
`runs/banking77-1500` to serve the larger model.

Useful local endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/systemone
```

Lev supports `choice`, `noul`, and `score` question shapes. The six-source
training path uses all three; `choice` is the easiest API path to understand
first.

## How does this work?

1. Banking77 provides a customer message and its correct intent label.
2. Lev turns that row into a request containing the message and all 77 intent
   options.
3. The tokenizer converts the request into token IDs.
4. Qwen reads the state and question. Its original weights stay mostly frozen;
   LoRA learns small updates to the transformer.
5. The pointer head compares the question's decision representation with each
   option representation and produces 77 scores.
6. Softmax converts the scores into probabilities. Python maps those positions
   back to the option names and returns JSON.

```text
JSON request → rendered text → tokens → Qwen + LoRA → pointer scores
            → softmax probabilities → JSON response
```

Qwen is not being trained from scratch, and it is not generating JSON. The
Python API wrapper guarantees the response shape; the model supplies the
probabilities.

## Stable benchmark

The repository contains a fixed 150-example Banking77 test suite:
[`benchmarks/banking77-v1`](benchmarks/banking77-v1).

Run it against a checkpoint with:

```bash
uv run python -m lev.benchmark \
  --run runs/banking77-1500 \
  --suite benchmarks/banking77-v1 \
  --out runs/benchmarks/banking77-v1
```

The first larger Lev run scored 88% accuracy on this suite. The benchmark also
reports NLL, Brier score, calibration error, confidence behavior, probability
normalization, latency, and checksums. Future model changes should be compared
against this same suite. The checked-in baseline is
[`baselines/lev-banking77-1500.json`](benchmarks/banking77-v1/baselines/lev-banking77-1500.json).

## Learn more

- [Learning Q&A](LEARNING_QA.md) — questions and answers from the build, in plain language.
- [Benchmarking guide](BENCHMARKING.md) — how to save results and iterate without overfitting the test set.
- [Model card](MODEL_CARD.md) — training recipe, results, limitations, and reproduction steps.
- [Goal and milestones](GOAL.md) — what Lev includes and what remains.
- [Kev's original prototype commit](https://github.com/jaredpalmer/kev/commit/d0e2b1fc4f9e410137b6b4ab7f7153fc52868a16).
- [Banking77 on Hugging Face](https://huggingface.co/datasets/legacy-datasets/banking77).

The Q&A covers tokenization, tensors, softmax, LoRA, the pointer head, labels,
attention masks, confidence, evaluation, benchmarks, and why a small model can
be confidently wrong.
