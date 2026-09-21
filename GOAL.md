# Lev Goal

Recreate the core idea of Kev with the smallest useful implementation.

## Core result

Given one customer message and a list of possible choices, return one probability per choice.

## Non-goals for now

Do not build yet:

- full TypeSafe API compatibility (`noul` and `score` remain later)
- six datasets
- production serving
- calibration systems
- Modal/GPU jobs
- policy-pair suites
- elaborate evaluation infrastructure
- `noul` and `score` question types

## Steps

1. Use Python 3.12 or 3.13 and `uv`.
2. Install the minimal packages: `torch`, `transformers`, `datasets==5.0.1`, `peft`, `accelerate`, and `pytest`.
3. Load Banking77 exactly like Kev with `load_dataset("legacy-datasets/banking77")`. ✅
4. Convert one row into a state, one choice question, 77 named options, and one correct label. ✅
5. Render that record into text. ✅
6. Load `Qwen/Qwen2.5-0.5B`. ✅
7. Add Kev's pointer head and block-causal mask. ✅
8. Produce one score and probability for every option. ✅
9. Train on a tiny sample until the model can overfit it.
10. Train rank-16 LoRA with Kev's target modules. ✅
11. Train a small Banking77 experiment. ✅
12. Evaluate accuracy and inspect predicted probabilities. ✅ (88.67% on 150 held-out rows)
13. Add multiple questions and Kev's question-isolation mask. ✅ (mask implemented and tested)
14. Add a TypeSafe-compatible `choice` input/output layer. ✅
15. Add `noul`, `score`, serving, and more datasets later.
16. Freeze and report a reproducible Banking77 benchmark. ✅ `benchmarks/banking77-v1`.

## Definition of done

Lev is successful when it can load Banking77 through Hugging Face, train on a tiny sample, and return probabilities for the 77 intents using Qwen plus a learned head and LoRA.

That first definition of done is now satisfied. The active next milestone is useful
held-out Banking77 accuracy from the 1,500-example run.

The first `choice` API milestone is also complete. `lev.serve` accepts
`POST /v1/systemone`, validates requests with Pydantic, runs the saved checkpoint,
and returns named probabilities. `noul`, `score`, and full SDK compatibility
remain future work.
