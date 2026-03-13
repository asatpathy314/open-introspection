# Prompt Tail Window Experiment

## Question

If prompt-side injection matters, how much of the prompt tail is actually
needed?

## Setup

- Script: `run_prompt_tail_window.py`
- Output: `results/prompt_tail_window.jsonl`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Vectors: 3.3 raw concept vectors
- Layer: `40`
- Prompt: `Tell me about today.`
- Concepts: first 10 Lindsey concepts
- Alpha: `8`
- Modes:
  - `baseline`
  - `assistant_only`
  - `tail_1`
  - `tail_2`
  - `tail_4`
  - `tail_8`
  - `prompt_all`
  - `all_positions`
- Samples: `1` generation per `(concept, mode)`
- Total rows: `80`

## Aggregate Results

| Mode | ID | Coherence | Mention |
|---|---:|---:|---:|
| `baseline` | 0.00 | 0.80 | 0.00 |
| `assistant_only` | 0.10 | 0.60 | 0.00 |
| `tail_1` | 0.40 | 0.70 | 0.20 |
| `tail_2` | 0.30 | 0.70 | 0.30 |
| `tail_4` | 0.60 | 0.70 | 0.40 |
| `tail_8` | 0.40 | 0.60 | 0.30 |
| `prompt_all` | 0.70 | 0.50 | 0.70 |
| `all_positions` | 0.80 | 0.50 | 0.50 |

## What The Results Say

- A short prompt-tail window can recover much of the open-ended steering
  effect.
- `tail_4` was the best short-window setting in this run: `0.60` ID accuracy
  with `0.70` coherence.
- `prompt_all` and `all_positions` were still strongest overall, but both had
  noticeably worse coherence than the better tail windows.
- The relationship was not monotonic. `tail_4` beat both `tail_2` and `tail_8`
  on judge ID.
- `assistant_only` remained weak again, which reinforces the position-scope
  result.

## Mechanistic Read

- The effect is not confined to a single token, but it also does not require
  prompt-wide injection to show up.
- A small prompt-tail intervention appears sufficient for many concepts.
- Since this run used only one generation per concept and mode, it should be
  treated as exploratory rather than definitive.

## Notable Oddity

- `dust` behaved strangely: it hit under `tail_1` and `tail_8`, but missed
  under both `prompt_all` and `all_positions`. That looks like a concept- and
  sample-specific instability rather than a clean monotonic effect.
