# Lev model card

## Model summary

Lev is an educational typed-decision model. It takes a state plus one or more
typed questions and returns a probability distribution over the supplied
answers. It does not generate prose or JSON tokens.

The reference model is a Qwen2.5-0.5B base model with rank-16 LoRA adapters, a
learned pointer head that scores each supplied option, and a block-causal mask
that isolates question branches while sharing the state.

Base model: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)

Repository: [`peterpme/lev`](https://github.com/peterpme/lev)

## Intended use

Lev is for learning, experimentation, and small local structured-classification
experiments. It is not a production financial decision system, and its public
Banking77 score must not be interpreted as accuracy on Backpack or any other
real product.

## Input and output

The public request contains a `state` and typed `questions`:

```json
{
  "state": "I need to exchange currencies in the app.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which intent best fits?",
      "criteria": {
        "exchange_via_app": "Exchange currencies in the app",
        "cash_withdrawal": "Withdraw cash"
      }
    }
  }
}
```

Lev returns one distribution per question. The Python wrapper guarantees the
JSON shape; the model supplies the probabilities.

## Training data

The primary experiment uses public Hugging Face datasets and no LLM-generated
training data:

| Source | Lev question type |
| --- | --- |
| Banking77 | Choice over 77 intents |
| BoolQ | Noul / yes-no |
| AG News | Choice plus derived yes-no questions |
| MNLI | Three-way Choice |
| SST-5 | Five-level Score |
| Yelp Review Full | Five-level Score plus yes-no |

Lev also contains optional loaders for TREC, DBpedia-14, IMDb, Amazon Reviews,
ARC-Challenge, OpenBookQA, CommonsenseQA, and Financial PhraseBank. Those
sources are separate experiments; more data is not automatically better because
different tasks can dilute the original Banking77 objective.

## Training procedure

Reference Banking77-only run:

| Setting | Value |
| --- | --- |
| Base | Qwen/Qwen2.5-0.5B |
| Examples | 1,500 Banking77 training rows |
| Epochs | 2 |
| LoRA rank | 16 |
| Gradient accumulation | 8 |
| Default learning rate | 2e-4 |
| Objective | Cross-entropy over the option distribution |

The six-source comparison uses 1,500 examples per source, two epochs, rank-16
LoRA, and the same pointer-head objective.

## Evaluation

Lev keeps two frozen suites in the repository:

- `banking77-v1`: 150 fixed Banking77 test records, used as the locked target
  regression test;
- `multi-source-v1`: 900 fixed held-out records, 150 from each of the six
  training sources, used to measure cross-task generalization.

Each suite has a manifest and SHA-256 checksum. The benchmark refuses to run if
the test bytes change. Reports include accuracy, NLL, Brier score, ECE,
high-confidence coverage and accuracy, latency, probability normalization, and
per-source accuracy/NLL.

For general-purpose progress, the repository also includes pinned Kev-compatible
fixtures under [`benchmarks/kev`](benchmarks/kev): `decision-v7` for development
selection, `transfer-v4` as a transfer guardrail, and `transfer-v9` as a newer
held-out transfer suite. These contain labelled TypeSafe-shaped requests, so
they test the same decision interface rather than only one dataset's label
names. Their locked test partitions are not for repeated hill-climbing.

### Reference results

| Model | Banking77-v1 | Multi-source-v1 |
| --- | ---: | ---: |
| Banking77-only | 88.0% | 45.1% |
| Six-source mixture | 86.7% | 74.7% |

The six-source mixture improves mixed-task accuracy by 29.6 percentage points
while losing 1.3 points on the original Banking77 suite. This is why Lev keeps
both suites: one measures the original skill, the other measures generalization.

The next generalist baseline should optimize development NLL on Kev's
`decision-v7`, keep accuracy and confidence-at-90% as guardrails, and require
no severe regression on `transfer-v4`. After selecting a candidate, run the
locked test partitions once and save the reports with the model card.

The durable experiment record is
[`multi-six-1500.json`](benchmarks/banking77-v1/experiments/multi-six-1500.json).

## How to reproduce

```bash
git clone https://github.com/peterpme/lev.git
cd lev
uv sync --extra serve

uv run python -m lev.train \
  --n_per_source 1500 --epochs 2 --accum 8 \
  --out runs/banking77-1500

uv run python -m lev.benchmark \
  --run runs/banking77-1500 \
  --suite benchmarks/banking77-v1 \
  --out runs/benchmarks/banking77-v1

uv run python -m lev.serve --run runs/banking77-1500 --port 8008
```

The first run downloads the base model and datasets from Hugging Face. Checkpoint
weights under `runs/` are intentionally not committed to this repository.

## Qwen2.5-specific implementation assumptions

Lev's model code is deliberately small and currently assumes Qwen2.5-like
interfaces:

- loading through `AutoModelForCausalLM`;
- accessing the transformer backbone through `.model`;
- reading `.last_hidden_state`;
- passing explicit `position_ids` and a four-dimensional attention mask;
- attaching LoRA to named `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj` modules;
- using existing Qwen tokenizer control tokens as internal delimiters.

That is what “written around Qwen2.5” means. The pointer head is mathematically
portable, but the backbone adapter is not guaranteed to be portable. Qwen3.5
uses a newer hybrid architecture and a vision-language interface, so a future
port would need to verify its backbone object, tokenizer/processor, attention
mask semantics, and LoRA target-module names before comparing scores. A Qwen3.5
result should be a separate benchmarked experiment, not a silent replacement of
the reference model.

## Limitations

- The model is small and can be confidently wrong.
- Accuracy depends on the supplied options and their wording.
- The public benchmark is educational and small, not production validation.
- The checkpoint is not currently hosted with this repository.
- Financial PhraseBank teaches sentiment, not Backpack transaction intent.
- Official TypeSafe SDK compatibility and production calibration are future work.

## Relationship to Kev

Lev mirrors the core educational mechanism in Kev. Kev’s model cards and README
document frozen suites, checksums, development/test separation, per-source
metrics, calibration, option-order tests, question-isolation tests, and recorded
configuration trials. Lev implements the smallest useful subset so each part can
be inspected locally.

- [Kev repository](https://github.com/jaredpalmer/kev)
- [Kev 0.5B model card](https://github.com/jaredpalmer/kev/blob/main/MODEL_CARD.md)
- [Kev benchmark README](https://github.com/jaredpalmer/kev/blob/main/README.md)
