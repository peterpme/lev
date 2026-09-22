# Lev research plan

This is the living experiment plan for Lev. It records what we are trying to
learn, which data and metrics are allowed to choose a model, and when we are
allowed to look at a locked test result. A finished experiment stays in the
repository as a result; we do not rewrite history to make a failed idea look
successful.

## Goal

Build the smallest useful typed-decision model and improve its generalization
over time. Lev should learn the decision interface and transfer across task
families, not merely memorize Banking77 intents.

The tiny-model focus is intentional:

- Lev: Qwen2.5-0.5B + pointer head + LoRA.
- Kev-0.5B: the closest published comparison because it also uses
  Qwen2.5-0.5B.
- Kev-0.6B: a useful small-model generation comparison, but it uses
  Qwen3-0.6B, so it is not a clean architecture-only comparison.
- Jev: hosted reference system, not local weights.

## What an eval suite is

An eval suite is a fixed exam for a model. Each JSONL record contains a state,
one or more typed questions, the available options, and a known answer. The
model never trains on the evaluation records.

The files are frozen so that a score means the same thing tomorrow as it does
today. If we edit the questions, labels, option order, or dataset revision,
the score is no longer directly comparable. We create a new suite version
instead of modifying an old one.

## Suites

| Suite | Role | Selection policy |
| --- | --- | --- |
| `decision-v4` | Historical tiny-Kev compatibility | Development during reproduction work |
| `decision-v7` | Current broad decision benchmark | Primary development selection |
| `transfer-v4` | New-task and held-out policy transfer | Generalization guardrail |
| `transfer-v9` | Harder transfer: MMLU-Pro, buried state, unknowable cases | Later robustness gate |
| `banking77-v1` | Original Lev regression check | Never use as the only objective |

Development may be read repeatedly while experimenting. Locked test partitions
are read once for a promoted candidate. The Kev-compatible fixtures and their
hashes live under [`benchmarks/kev`](benchmarks/kev).

## Comparison matrix

Every system must receive the same request bytes and be scored by the same
semantic labels. The raw Qwen base cannot be compared as if it were a decision
model: it has no pointer head and does not naturally return Lev JSON.

| System | Backbone | Decision readout | What it tells us |
| --- | --- | --- | --- |
| Qwen2.5 base probe | Qwen2.5-0.5B | Next-token/letter probe | What the untrained language model knows; diagnostic only |
| Qwen2.5 readout-only | Qwen2.5-0.5B frozen | Train pointer head, no LoRA | Value of the readout alone |
| Lev | Qwen2.5-0.5B | Pointer head + LoRA | Current Lev system |
| Kev-0.5B | Qwen2.5-0.5B | Kev pointer head + LoRA | Closest published comparison |
| Kev-0.6B | Qwen3-0.6B | Kev pointer head + LoRA | Small newer-backbone comparison |
| Jev | Hosted | Hosted decision system | External reference, not a controlled training ablation |

The most important comparison is Lev versus Kev-0.5B on the same `decision-v4`
and `transfer-v4` items. Kev-0.6B and Jev are useful context, but differences
in backbone, training recipe, and possible data exposure must be stated.

## Registered experiment sequence

### Phase 1: establish tiny baselines

1. Train Lev on the same ten public source families used by current Kev where
   Lev has loaders.
2. Evaluate Lev on `decision-v4` and `transfer-v4` development.
3. Load `jaredpalmer/kev-0.5b` and `jaredpalmer/kev-0.6b` through Kev's own
   evaluator on the same development suites.
4. Run the hosted Jev comparison on the same development records if the Jev
   credentials are available.
5. Save one result JSON per system, including base revision, data sources,
   seed, code commit, suite manifest hash, and dependency lock hash.

### Phase 2: add the missing evaluator checks

Implement the small but important checks that ordinary accuracy misses:

- option-order permutation flip rate;
- question isolation when multiple questions share a state;
- paired minimal examples where one decisive fact changes;
- confidently-wrong rate at probability ≥ 0.9;
- coverage at a fixed error budget;
- unknowable examples where the deciding evidence is absent.

These checks should be paired by record or minimal-example group, not treated
as unrelated rows.

### Phase 3: hill-climb one knob at a time

Change one variable, keep the data and suites fixed, and register the expected
outcome before training:

- learning rate;
- LoRA targets or rank;
- number of epochs;
- source mixture and per-source count;
- option permutation and distractor augmentation;
- synthetic policy/compositional examples;
- base model.

Primary selection metric: macro development NLL across source families.
Guardrails: transfer accuracy, Brier score, ECE, confident errors, and
original-task retention. If a candidate wins only Banking77 and loses transfer,
it is not a generalist improvement.

### Phase 4: Qwen3.5 port

Only after the Qwen2.5 baseline is stable, port the Lev adapter to
`Qwen3.5-0.8B-Base`. Keep the data, seed, epochs, LoRA rank, and benchmark
bytes fixed. Verify hidden states, tokenizer/processor behavior, LoRA target
names, and the attention-mask contract before training. Treat the result as a
new model family, not a silent replacement.

## Jev comparison boundary

Jev evaluations live partly outside the local repository because Jev is a
hosted service. Kev's `kev.jev` helper sends the same frozen records through
the Vercel AI Gateway and saves the responses. We can record those outputs in
Lev as an external-reference result, but we should not use them as training
labels or pretend the comparison isolates model architecture: Jev's training
data is not controlled by this experiment.

## Result log

| Date | System | Suite | Dev result | Decision |
| --- | --- | --- | --- | --- |
| — | — | — | — | First generalist baseline pending |

