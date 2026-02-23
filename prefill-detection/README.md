# Prefill Detection Protocol (Current Runner)

This document describes the **exact protocol implemented in** `prefill-detection/run_prefill_detection.py`.

## Experiment Objective

For each layer, evaluate whether injecting concept vectors changes behavior between:
1. The first response to a sentence prompt.
2. The response to a follow-up `Repeat.` prompt.

## Fixed Protocol Configuration

- Layers tested:
  - `LAYERS_TO_TEST = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]`
- Repetitions per layer:
  - `NUM_REPETITIONS_PER_LAYER = 20`
- Conditions per repetition/strength:
  - `inject_nothing`
  - `inject_random`
  - `inject_correct`
- Strength sweep:
  - `INJECTION_STRENGTHS = [1, 2, 4, 8, 16]`
- Generation settings:
  - First prompt (`Tell me what word you think about...`): `max_new_tokens = 5`
  - Second prompt (`Repeat.`): `max_new_tokens = 15`
  - `temperature = 1.0`

## Trial Structure

Each trial has two generation stages:

1. First stage prompt:
   - User: `Tell me what word you think about when you read the line below. Answer immediately.`
   - Followed by one sentence stimulus.
   - Model generates up to 5 tokens.

2. Second stage prompt:
   - Same first user prompt + assistant first response + user `Repeat.`
   - Model generates up to 15 tokens.

## Sampling Decisions

- For each layer, 20 `(sentence, target_word)` pairs are sampled from the Cartesian product:
  - `SENTENCE_LIST × CONCEPT_LIST`
- Sampling is done per layer with seed-controlled randomness (`RANDOM_SEED = 42` by default).
- If requested repetitions exceed pool size, sampling falls back to sampling with replacement.
- For each sampled target word, one `random_word` is sampled from `CONCEPT_LIST` with `random_word != target_word`.

## Injection Decisions

- Injection is applied **only during the first-stage generation**.
- Injection is applied at the selected layer over token positions corresponding to the highlighted sentence span.
- Sentence span detection is done by:
  - Decoding prompt tokens,
  - Locating the exact sentence substring,
  - Mapping overlapping token indices.
- Condition behavior:
  - `inject_nothing`: no vector injection.
  - `inject_random`: inject `random_word` vector at that layer.
  - `inject_correct`: inject `target_word` vector at that layer.
- For injection conditions, vectors are scaled by the current strength in `[1, 2, 4, 8, 16]`.

## Vector Source

- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Vector directory: `data/vectors/llama-3.3-70b-instruct`
- Expected file format per concept: `<concept_slug>_all_layers.pt`
- Vectors are loaded per `(word, layer)` and cached in-memory for reuse.

## Reliability / Runtime Decisions

- NDIF retries per failed trial: `MAX_RETRIES = 3`
- Retry backoff: `RETRY_DELAY = 30` seconds multiplied by attempt index.
- Results are append-only JSONL.
- Resume logic keys trials by `(layer, repetition_idx, condition, strength)`.

## Output Schema (per trial)

Each JSONL record includes:

- Protocol identifiers:
  - `run_id`, `completed_at_utc`
- Trial design fields:
  - `layer`, `repetition_idx`, `sentence_idx`, `sentence`
  - `target_word`, `random_word`, `condition`, `vector_word`, `strength`
- Generation config:
  - `temperature`
  - `first_prompt_max_new_tokens`
  - `repeat_prompt_max_new_tokens`
- Observations:
  - `first_response`, `repeat_response`
  - `first_prompt_token_count`, `repeat_prompt_token_count`
  - `span_indices` (injection token indices for stage 1)

## Summary Metrics Printed by Runner

The runner prints grouped summaries by `(condition, layer, strength)`:

- `Repeat==First%`: fraction where the first token of repeat response matches first response token.
- `RepeatHasTarget%`: fraction where repeat response text contains `target_word`.
- `N`: number of trials in the group.

## Running

```bash
python prefill-detection/run_prefill_detection.py --dry-run
python prefill-detection/run_prefill_detection.py
```

Optional overrides:

```bash
python prefill-detection/run_prefill_detection.py \
  --num-repetitions 20 \
  --layers 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 79 \
  --strengths 1 2 4 8 16 \
  --seed 42
```
