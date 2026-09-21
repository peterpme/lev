# Lev learning checkpoint: simple Q&A

This document records the core ideas discussed while recreating Jared Palmer's
Kev prototype. The reference point is Kev commit
`d0e2b1fc4f9e410137b6b4ab7f7153fc52868a16`.

The first milestone is Banking77 only. Given one banking message and 77
possible banking intents, Lev predicts which intent is the best answer.

## What are we building?

Lev is a small version of Kev:

```text
customer message + possible choices
                ↓
Qwen + LoRA + pointer head
                ↓
one probability per choice
```

We are not pretraining Qwen. Qwen is already pretrained. We are adapting it to
understand a decision format and score supplied choices.

## Where do Qwen and Banking77 come from?

Both are public Hugging Face resources:

- `Qwen/Qwen2.5-0.5B` is the pretrained language model and tokenizer.
- `legacy-datasets/banking77` is the labelled Banking77 dataset.

Banking77 was created by PolyAI. It contains 13,083 English banking questions
across 77 intents. The raw data contains customer text and an integer label.

Lev loads it with the same call as Kev:

```python
load_dataset("legacy-datasets/banking77", split="train")
```

The dataset is Parquet-backed, which means Hugging Face stores it in a modern
columnar data format. It does not mean the contents are different from rows of
text and labels.

## What does a Banking77 row look like?

Conceptually:

```json
{
  "text": "How do I exchange currencies with this?",
  "label": 31
}
```

The integer label is an ID, not a sentence. The dataset metadata maps it to an
intent name:

```text
31 → exchange_via_app
```

The model does not read the number `31` as text. Lev looks up the intent name
and uses that name to construct the choices.

There are two labels to distinguish:

```text
Banking77 label:       31 → exchange_via_app
Lev request label:      7 → exchange_via_app is seventh after shuffling
```

The second value is the option position used for that particular presentation.

## Where does the decision request come from?

During training, Lev generates it from each Banking77 row. For production, the
caller sends it as JSON.

Lev turns a raw row into a request like:

```json
{
  "state": "How do I exchange currencies with this?",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which banking intent best describes this customer message?",
      "criteria": {
        "activate_my_card": "Customer asks about activate my card",
        "exchange_via_app": "Customer asks about exchange via app",
        "cash_withdrawal": "Customer asks about cash withdrawal"
      },
      "label": "exchange_via_app"
    }
  }
}
```

The real request has all 77 intent options. The example is shortened.

## Is that JSON the TypeSafe API?

Yes. Kev's `api.py` defines the public TypeSafe-compatible request and response
contract. It validates three question types:

```text
noul   → yes/no
choice → select one named option
score  → select an ordered level
```

The endpoint in Kev is:

```text
POST /v1/systemone
```

Lev currently skips this public API layer. It constructs equivalent internal
Python dictionaries directly so we can learn the model first. Adding the API
later will not change the neural model; it will add validation, JSON parsing,
and response formatting around the model.

## Does the JSON become text?

Yes, after validation and rendering. The model does not receive a Python dict or
a Markdown file. It ultimately receives token IDs.

The JSON is rendered into a text-like internal sequence with special markers:

```text
<|fim_prefix|>document: How do I exchange currencies with this?
<|fim_middle|>Which banking intent best describes this customer message?
<|box_start|>activate_my_card<|box_end|>
<|box_start|>exchange_via_app: Customer asks about exchange via app<|box_end|>
<|box_start|>cash_withdrawal: Customer asks about cash withdrawal<|box_end|>
...
<|fim_suffix|>
```

This is a conceptual display. The actual model input is an integer token array,
plus position IDs, segment IDs, and an attention mask.

An inspected Lev example contained 77 options and became 658 tokens. The first
token IDs corresponded to:

```text
<|fim_prefix|>, document, :, How, do, I, query, a, payment, ...
```

Qwen's tokenizer can split one word into multiple subword tokens. For example,
`activate_my_card` became pieces resembling:

```text
activate, _my, _card
```

## What are the special tokens?

Lev reuses special tokens already in Qwen:

```text
<|fim_prefix|>  shared state begins
<|fim_middle|>  question begins
<|box_start|>   option begins
<|box_end|>     option ends
<|fim_suffix|>  decision position
```

They are structural markers. They tell the model where the state, question,
options, and decision representation are located.

## What does “Qwen transformer backbone only” mean?

A text-generation model has two broad parts:

```text
transformer layers → vocabulary/output head
```

The transformer layers turn token sequences into contextual hidden vectors. The
vocabulary head turns a hidden vector into scores for the next word/token.

Lev does not generate text. It needs the contextual vectors, not next-word
prediction. Therefore it keeps Qwen's transformer layers and discards the
vocabulary/output head, then attaches its own pointer head.

```text
Qwen transformer → Lev pointer head → option probabilities
```

## Why is the state limited to 384 tokens?

A token is a subword unit, not necessarily a whole word. Lev allows at most 384
tokens for the shared state, including its marker. This is a practical budget
copied from Kev's setup, not a magical property of Qwen.

The budget prevents a huge state from consuming the entire context window and
keeps attention and memory manageable. The total state-plus-question branch
budget is approximately 1,024 tokens in the training path.

## What is ordinary causal attention?

Causal attention means a token can read earlier tokens but not future tokens.
It is the normal left-to-right attention pattern used by autoregressive language
models.

For example, when processing the word `currencies`, the model can see words
before it in the sequence, but not later words.

## What is the block-causal mask?

Lev may pack several questions into one sequence:

```text
shared state
question 1 + options + decision 1
question 2 + options + decision 2
```

Each token receives a segment ID:

```text
0 = shared state
1 = question 1
2 = question 2
```

The mask permits a question to read:

- earlier shared-state tokens;
- earlier tokens in its own question;
- not tokens belonging to another question.

So question 2 can read state and question 2, but not question 1. This prevents
one question from leaking information into another while still allowing the
state to be shared efficiently.

## What is a pointer?

A pointer selects one item from a supplied list.

A fixed classifier always has the same output classes:

```text
class 0, class 1, ..., class 76
```

Lev instead receives options in the request and scores those options directly:

```text
decision representation + option 1 representation → score 1
decision representation + option 2 representation → score 2
...
```

With 77 Banking77 options, it produces 77 scores. With a production request
containing 3 choices, it produces 3 scores.

## How does the pointer head work?

Qwen's hidden size is 896. Lev takes:

- the hidden vector at `<|fim_suffix|>` as the decision vector;
- the hidden vector at each `<|box_end|>` as an option vector.

The pointer head projects both kinds of vectors to 256 dimensions:

```text
decision: 896 → 256
option:   896 → 256
```

It computes a scaled dot product between the decision projection and every
option projection. That creates one logit per option. Softmax converts the
logits into probabilities.

## What is LoRA?

Full fine-tuning would update all of Qwen's parameters. Freezing Qwen entirely
would train only the new pointer head.

LoRA is the middle path. It freezes each original weight matrix `W` and learns a
small low-rank update:

```text
effective weight = W + scale × (A × B)
```

Lev uses rank 16, alpha 32, and dropout 0.05. Only the small adapter matrices
and pointer head are trainable. The base Qwen weights stay in the Hugging Face
cache.

## What are the LoRA targets?

The target names identify the Qwen linear layers where adapters are inserted:

```text
q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj
```

The first four are attention projections. The last three are feed-forward/MLP
projections. “Targeting” them means attaching trainable LoRA updates to those
layers; it does not mean selecting data labels.

Lev trains approximately 9.26 million parameters instead of updating all Qwen
parameters. A checkpoint stores a roughly 34 MB adapter and a roughly 1.8 MB
pointer head.

## How does presentation variation work?

Kev/Lev changes the appearance of the same underlying task so the model cannot
memorize one template.

The state may be:

```text
plain customer text
document: customer text
ticket: channel + body
chat list: role + content
```

The instruction is usually a string, but sometimes becomes a small structured
object. Each option description is independently removed about half the time.
The option order is shuffled each epoch.

Sometimes the correct option is removed and replaced with:

```text
other: None of the above
```

Sometimes an unrelated distractor such as `pancakes` is added. These are
synthetic augmentations intended to teach robustness to changing option lists.

The meaning of the Banking77 message is not changed except in the intentional
“none of the above” augmentation.

## What is a smoke run?

A smoke run is a tiny end-to-end plumbing test, not a serious model-quality
experiment.

It checks that we can:

1. Download Banking77.
2. Download Qwen.
3. Construct a 77-option record.
4. Tokenize it.
5. Run Qwen and the pointer head.
6. Produce 77 probabilities.
7. Backpropagate through LoRA.
8. Save a checkpoint.
9. Reload the checkpoint.
10. Evaluate test examples.

Our 40-row smoke model scored 0/40 on held-out examples. That was expected: it
only saw 40 training messages for 77 intents. The run proved the machinery
worked, not that the model was useful.

## What does accuracy measure?

For each held-out example, Lev selects the option with the largest probability.
Accuracy is:

```text
number of exact correct choices / number of evaluated examples
```

If 63 of 150 test rows are correct, accuracy is 42%. With 77 choices, random
guessing is about 1.3% accuracy.

Evaluation uses Banking77's official test split, not the training rows used to
update the model.

## What is NLL?

NLL means negative log-likelihood. For one example:

```text
NLL = -log(probability assigned to the correct option)
```

Assigning probability 1.0 to the correct answer gives NLL 0. Assigning a tiny
probability to the correct answer creates a large penalty. Therefore NLL catches
confident wrong predictions that accuracy alone treats as simply “wrong.”

## What does the evaluation command do?

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python -m lev.evaluate \
  --run runs/banking77-1500 \
  --n 150
```

It loads the original cached Qwen model, the saved LoRA adapter, and the saved
pointer head. It then loads 150 examples from Banking77's test split, recreates
the 77-option request, runs predictions, calculates accuracy/mean confidence/NLL,
and writes `runs/banking77-1500/eval.json`.

The environment variable prevents a stale saved Hugging Face login token from
being used implicitly. The model and dataset are public, so no login token is
needed.

## What is standard and what is custom?

Standard reusable pieces:

- Hugging Face Datasets downloads and reads Banking77.
- Transformers loads Qwen and its tokenizer.
- PEFT implements LoRA.
- PyTorch provides tensors, modules, optimizers, and loss functions.
- scikit-learn or other metric libraries could calculate accuracy.

Kev-specific glue:

- turning Banking77 into a choice request;
- special-token packing;
- block-causal question isolation;
- the variable-option pointer head;
- mapping probabilities back to named JSON answers.

We could replace Lev with a normal 77-class classifier, a Sentence Transformers
similarity model, or a cross-encoder. Those would be legitimate alternatives,
but they would no longer mirror Kev's variable-choice pointer architecture.

## What is currently implemented?

- `lev/data.py`: Hugging Face loading, request construction, rendering, and augmentation.
- `lev/model.py`: token packing, mask, Qwen backbone, LoRA, and pointer head.
- `lev/train.py`: loss, optimizer, scheduler, MPS training, and checkpoint saving.
- `lev/evaluate.py`: checkpoint reload and held-out evaluation.
- `lev/render.py`: inspect a transformed example before tokenization.
- `scripts/train-background.sh`: detached tmux training launcher.
- `LOG.md`: chronological implementation log.
- `GOAL.md`: remaining roadmap.

The TypeSafe JSON API, `noul`, `score`, serving, calibration, and Kev's other
five datasets are intentionally deferred until the Banking77 `choice` path is
understood.

## How do we add the TypeSafe API later?

The next API layer would:

1. Add Pydantic models for `SystemOneRequest`, `Choice`, `Noul`, and `Score`.
2. Validate incoming JSON.
3. Convert JSON values into the internal rendered record Lev already uses.
4. Run the current tokenizer/model/pointer path.
5. Map option positions back to names.
6. Return typed JSON with the chosen option, confidence, and probabilities.
7. Expose that function through `POST /v1/systemone`.

The neural model does not need to change for the first `choice` API milestone.

## What happened in the first meaningful run?

The 1,500-example, two-epoch Banking77-only run completed 376 optimizer
updates on Apple MPS. On 150 unseen Banking77 test rows it achieved:

```text
accuracy:        88.67%
mean confidence: 88.95%
NLL:             0.51
```

This is a promising first result, not a final benchmark. The evaluation sample
was only 150 test rows and the model has not yet been trained on Kev's other
datasets.

## How was this code written?

The implementation was reconstructed from Kev's original source, dependency
lockfile, and behavior. Three independent audits checked the original model,
data pipeline, and minimal trainer. The remaining pieces follow standard
PyTorch/Hugging Face patterns, while the Kev-specific packing/masking/pointer
logic was copied in simplified form.

This is why Lev is small: it uses existing libraries for the hard general
infrastructure and only writes the custom decision-model glue.
