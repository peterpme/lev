# Lev build log

Lev is a learning-sized reimplementation of the first Kev prototype. The source of truth is Jared Palmer's Kev commit `d0e2b1fc4f9e410137b6b4ab7f7153fc52868a16`.

The narrow first milestone is:

```text
Banking77 customer message + 77 named intents
                    ↓
Qwen hidden states + pointer head
                    ↓
77 scores → softmax → 77 probabilities
```

We are **not** training a language model from scratch. Qwen already knows how to turn text into useful internal vectors. Lev teaches a small adaptation of Qwen, plus a new head, how to compare one decision with a supplied list of choices.

## 1. What was copied from Kev

We inspected Kev's first commit directly rather than guessing from the current README. Lev currently mirrors these decisions:

- Base model: `Qwen/Qwen2.5-0.5B`.
- Dataset loader: `load_dataset("legacy-datasets/banking77", split=...)`.
- Qwen transformer backbone only; the vocabulary/output generation head is discarded.
- Qwen's existing special tokens delimit state, question, options, and decision.
- State limit: 384 tokens. Total state-plus-branch limit: 1,024 tokens.
- A block-causal mask isolates question branches while allowing each branch to read shared state.
- Pointer projections: Qwen hidden size 896 → pointer size 256.
- Rank-16 LoRA, alpha 32, dropout 0.05.
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- AdamW, learning rate `2e-4`, weight decay `0.01`, gradient clipping at `1.0`, and OneCycleLR.
- Fresh option-order/data augmentation each epoch.
- Checkpoints contain the LoRA adapter, pointer head, tokenizer, and metadata—not a duplicate copy of all Qwen weights.

Lev intentionally omits Kev's other five datasets, TypeSafe API, `noul`, `score`, serving, calibration, permutation-KL regularization, and ordinal loss for this first batch.

## 2. Project and dependencies

Created `/Users/peter/Projects/lev`, initialized Git, and installed the environment with:

```bash
uv sync --python 3.13
```

Resolved direct packages:

```text
accelerate      1.15.0
datasets        5.0.1
numpy           2.5.3
peft            0.21.0
scikit-learn    1.9.1
torch           2.8.0
transformers    4.57.6
pytest          9.1.1
```

Except for pytest, those are the exact versions resolved by Kev's original `uv.lock`.

### Correction: the CSV detour was unnecessary

An early attempt assumed modern `datasets` could no longer load Kev's legacy Banking77 identifier, so we briefly staged CSV/JSONL files. That diagnosis was wrong. Kev's own lockfile uses `datasets==5.0.1`, and the exact call still works because the Hub copy is Parquet-backed.

The misleading `DatasetNotFoundError` came from stale implicit Hugging Face authentication plus cache permissions on this machine. Public loading succeeds with:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

The obsolete staged data files were moved to Trash. Training now downloads through Hugging Face and transforms examples in memory, just like Kev.

## 3. Where Banking77 and Qwen come from

They are two different Hugging Face repositories:

- `legacy-datasets/banking77` supplies labelled examples. Each row has customer text and an integer label; the dataset feature metadata maps that integer to one of 77 intent names.
- `Qwen/Qwen2.5-0.5B` supplies a pretrained tokenizer and roughly 0.5-billion-parameter language model.

No Hugging Face token is required for either public repository. Files download once and are cached locally.

Jared did not invent or hand-label Banking77. PolyAI published the underlying dataset: 13,083 online-banking queries grouped into 77 intents, with 10,003 train rows and 3,080 test rows. Jared's contribution was converting those existing labelled rows into Kev's decision format and combining them with other public datasets.

## 4. The Banking77 transformation

The Hub gives us a row conceptually like:

```json
{
  "text": "How do I exchange currencies with this",
  "label": 31
}
```

The dataset's metadata tells us that label 31 names an intent such as `exchange_via_app`. Lev turns it into a labelled request:

```json
{
  "state": "How do I exchange currencies with this",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which banking intent best describes this customer message?",
      "criteria": {
        "activate_my_card": "Customer asks about activate my card",
        "exchange_via_app": "Customer asks about exchange via app"
      },
      "label": "exchange_via_app",
      "src": "banking77"
    }
  }
}
```

The real `criteria` object has all 77 intents; the example is shortened for readability. `materialize()` preserves their order and converts the named gold label to its numeric position in that particular order.

Kev deliberately varies presentation so the model learns the task instead of memorizing one template:

- A description template is chosen per record.
- Each option description is independently removed 50% of the time.
- Instructions become a small structured object 15% of the time.
- State is sometimes wrapped as a document, support ticket, or chat message.
- Options are shuffled every epoch.
- 10% of eligible choice questions remove the correct answer and make `other: None of the above` correct.
- Otherwise, about 13.5% receive one obviously unrelated distractor.

This happens in memory in `lev/data.py`. JSONL output is optional and only useful for inspection.

## 5. Tokenization and packing

A tokenizer does not understand meaning by itself. It converts text into integer token IDs that index Qwen's embedding table. For example, a phrase may become several subword IDs rather than one ID per word.

Lev reuses five special tokens already present in Qwen, so it does not need to add or train new embedding rows:

```text
<|fim_prefix|>  begins shared state
<|fim_middle|>  begins a question
<|box_start|>   begins an option
<|box_end|>     ends an option
<|fim_suffix|>  asks for the decision
```

One packed record is shaped like:

```text
<state> customer message
<question> instructions
<option> option 1 </option>
<option> option 2 </option>
...
<option> option 77 </option>
<decide>
```

`encode()` records three parallel arrays:

- `ids`: token IDs fed into Qwen.
- `seg`: 0 for shared state and 1, 2, ... for question branches.
- `pos`: token positions, restarted after state for each question branch.

It also records the hidden-state index of `<decide>` and every `</option>` token. Caller text matching a Qwen control-token spelling is rewritten before tokenization, so user data cannot forge option boundaries.

Our first real example packed to 658 tokens.

## 6. The block-causal mask

Ordinary causal attention lets a token read earlier tokens. Kev adds a segment rule:

```text
allow(i, j) = j <= i AND (segment[j] == 0 OR segment[j] == segment[i])
```

In plain English, a question can read shared state and itself, but it cannot peek into sibling questions. Packing multiple questions therefore saves repeated state computation without allowing answers to leak between branches.

The Banking77 milestone has one question, but Lev implements and unit-tests this mask now because it is central to Kev's design.

## 7. How the pointer head makes 77 scores

Qwen2.5-0.5B produces one 896-number hidden vector for every token. Lev takes:

- the 896-number vector at `<decide>`;
- one 896-number vector at each option's closing token.

The pointer head contains two learned linear projections:

```text
decision vector: 896 → 256  (query)
option vector:   896 → 256  (key)
```

For each option it computes:

```text
score(option) = dot(key(option), query(decision)) / sqrt(256)
```

Seventy-seven option vectors therefore produce 77 scores. Softmax converts those arbitrary scores into 77 positive probabilities that sum to 1. This is not JSON generation and it is not a fixed 77-class classifier: the number of scores follows the number of options supplied in the request.

## 8. What LoRA changes

Freezing Qwen completely would leave only the new pointer head trainable. Full fine-tuning would update every Qwen parameter and require much more memory and checkpoint space.

LoRA is the middle path. For selected Qwen matrices, it freezes the original matrix `W` and learns two much smaller rank-16 matrices whose product is an update:

```text
effective weight = W + scale × (A × B)
```

Rank 16 is tiny compared with Qwen's normal matrix dimensions. The original language knowledge stays in frozen Qwen weights while the adapters learn how Kev's delimiters and decision task should behave.

Lev trains 9.26 million parameters total: the rank-16 adapters plus the pointer head. A tiny checkpoint showed the consequence:

```text
LoRA adapter: 34 MB
pointer head:  1.8 MB
```

The base Qwen model remains in the Hugging Face cache and is combined with these files when loading Lev.

## 9. Training mechanics

For each example:

1. Shuffle/augment its options.
2. Materialize the correct option's current index.
3. Tokenize and pack the record.
4. Run Qwen with the block-causal mask.
5. Produce one score per option with the pointer head.
6. Compare scores with the correct index using cross-entropy loss.
7. Backpropagate gradients into the pointer head and LoRA parameters.
8. Accumulate several records before an optimizer update.
9. Clip gradients, update parameters with AdamW, and advance the learning-rate schedule.

The first GPU dry run on Apple MPS produced:

```json
{
  "sequence_tokens": 658,
  "options": 77,
  "probability_sum": 1.0000001192092896,
  "label": 63
}
```

That proves the entire untrained inference path produces a valid distribution over all 77 choices.

The 12-record end-to-end training proof completed three optimizer updates on MPS:

```text
update 1/3  loss 13.5877
update 2/3  loss 13.5255
update 3/3  loss 12.9972
```

Loss moving down over only three updates is encouraging, but this run exists to prove gradients and checkpoint saving—not model quality.

### Small-run scheduler fix

Kev uses `OneCycleLR(..., pct_start=0.1)`. With exactly 10 optimizer updates, PyTorch 2.8 creates a zero-length warmup phase and divides by zero. Kev's full training has thousands of updates, so it does not encounter this edge case.

Lev keeps Kev's 10% warmup for runs above 10 updates and uses 30% for tiny runs. A regression test covers the boundary.

## 10. Background execution

The launcher uses a detached tmux session so training survives after the launch command exits:

```bash
./scripts/train-background.sh
```

It writes:

```text
runs/banking77-smoke/train.log
runs/banking77-smoke/train.pid
runs/banking77-smoke/train.session
```

Watch progress with:

```bash
tail -f runs/banking77-smoke/train.log
```

The first `nohup` implementation was reaped by the automation shell before Python started. That was a launcher issue, not a training failure; tmux makes the process lifetime explicit and inspectable.

The corrected 40-record smoke run completed 10 updates on MPS at about 0.37
seconds per record and saved `runs/banking77-smoke`.

Reloading that checkpoint and evaluating 40 held-out test rows produced:

```json
{
  "examples": 40,
  "accuracy": 0.0,
  "mean_confidence": 0.5883798964321614,
  "nll": 11.30124251824299
}
```

Zero correct is unsurprising here: random chance is about 1/77 (1.3%), and the
model saw only 40 total training messages for one epoch. This result validates
checkpoint loading and evaluation, not model quality. It also demonstrates why
confidence must eventually be calibrated—the tiny model is confidently wrong.

The first meaningful Banking77-only run was then launched with the same 1,500
examples per source used in Kev's released recipe:

```bash
LEV_N_PER_SOURCE=1500 LEV_EPOCHS=2 LEV_ACCUM=8 \
  ./scripts/train-background.sh runs/banking77-1500
```

It runs in tmux session `lev-banking77-1500` and logs to
`runs/banking77-1500/train.log`.

That run completed 376 optimizer updates and saved its checkpoint. On 150
unseen Banking77 test examples, the first meaningful result was:

```json
{
  "accuracy": 0.8866666666666667,
  "mean_confidence": 0.8894553456703822,
  "nll": 0.5103092139632985
}
```

This is a promising first experiment, not a final benchmark: the evaluation
sample is only 150 rows and the run used Banking77 alone.

## 11. Verification so far

- Exact Kev dependency versions resolved.
- Exact Hugging Face Banking77 loader succeeds.
- Qwen2.5-0.5B downloads and loads.
- Unit tests cover data materialization, augmentation, structured rendering, pointer output shape, question isolation, and the tiny scheduler boundary.
- Six tests pass on Python 3.13.
- CPU and Apple MPS forward passes produce 77 normalized probabilities.
- Apple MPS backward pass, optimizer updates, and checkpoint save succeed.
- A detached Banking77 smoke run can be launched and monitored.
- A saved checkpoint reloads and evaluates against the Banking77 test split.

## 12. What remains after this first batch

The next useful steps are intentionally sequential:

1. Reload a saved checkpoint and evaluate held-out Banking77 accuracy.
2. Overfit a tiny fixed set as a stronger learning sanity check.
3. Train on more Banking77 rows and inspect mistakes/probability confidence.
4. Add Kev's other public datasets one at a time: BoolQ, AG News, MNLI, SST-5, and Yelp Review Full.
5. Add `noul` and ordered `score` questions when their datasets require them.
6. Add TypeSafe-compatible request/response models only after the learning loop is understood.

## Key files

- `lev/data.py`: Hugging Face loading, Kev-style transformation, and augmentation.
- `lev/model.py`: tokenizer packing, block-causal mask, Qwen backbone, LoRA, and pointer head.
- `lev/train.py`: loss, optimizer, scheduler, MPS training, and checkpoint saving.
- `lev/api.py`: Pydantic request/response models and JSON/tensor conversion.
- `lev/serve.py`: choice-only FastAPI server.
- `lev/render.py`: inspect a transformed row before tokenization.
- `scripts/train-background.sh`: detached training launcher.
- `tests/`: small tests for each core mechanism.

## External references

- Kev source commit: <https://github.com/jaredpalmer/kev/commit/d0e2b1fc4f9e410137b6b4ab7f7153fc52868a16>
- Qwen model: <https://huggingface.co/Qwen/Qwen2.5-0.5B>
- Banking77 Hub copy: <https://huggingface.co/datasets/legacy-datasets/banking77>
- Banking77 paper: <https://arxiv.org/abs/2003.04807>

## 2026-09-21 — choice-only FastAPI layer

Added the first public API boundary:

- `lev/api.py` defines Pydantic request/response models, JSON rendering, and
  tensor-to-named-answer conversion.
- `lev/serve.py` loads the base model, LoRA adapter, pointer head, and tokenizer
  once at startup and exposes `POST /v1/systemone`.
- `uv sync --extra serve` installs FastAPI and Uvicorn.
- Nine tests pass, including request validation and response mapping.

A live MPS request returned `exchange_via_app` from a hand-written three-option
JSON request in approximately 492 ms. The endpoint currently supports `choice`
only; `noul`, `score`, and official SDK compatibility are intentionally next.
