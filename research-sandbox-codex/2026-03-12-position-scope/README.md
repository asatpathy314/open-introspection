# Position Scope Experiment

## Question

Does open-ended steering on Llama-3.3-70B-Instruct depend mainly on where the
vector is injected?

## Setup

- Script: `run_position_scope.py`
- Output: `results/position_scope_openended.jsonl`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Vectors: 3.3 raw concept vectors
- Layer: `40`
- Prompt: `Tell me about today.`
- Concepts: first 10 Lindsey concepts
- Scopes:
  - `assistant_only`
  - `prompt_only`
  - `last_prompt_token`
  - `all_positions`
- Alphas: `0`, `8`
- Samples: `2` generations per `(concept, scope, alpha)`
- Total rows: `160`

## Aggregate Results

Metrics are judge identification accuracy, coherence rate, and concept mention
rate.

| Scope | Alpha | ID | Coherence | Mention |
|---|---:|---:|---:|---:|
| `assistant_only` | 0 | 0.10 | 0.70 | 0.00 |
| `assistant_only` | 8 | 0.20 | 0.85 | 0.00 |
| `last_prompt_token` | 0 | 0.10 | 0.85 | 0.00 |
| `last_prompt_token` | 8 | 0.45 | 0.75 | 0.20 |
| `prompt_only` | 0 | 0.10 | 0.90 | 0.00 |
| `prompt_only` | 8 | 0.60 | 0.55 | 0.40 |
| `all_positions` | 0 | 0.25 | 0.80 | 0.00 |
| `all_positions` | 8 | 0.80 | 0.60 | 0.40 |

## What The Results Say

- `assistant_only` was weak. Adding the vector only after the assistant starts
  generating barely moved judge accuracy.
- Prompt-side injection mattered a lot on 3.3. Both `prompt_only` and
  `last_prompt_token` beat `assistant_only`.
- `all_positions` was strongest overall on ID accuracy, but it also reduced
  coherence.
- `prompt_only` was stronger than the early partial run suggested. On this
  completed run, it reached `0.60` ID accuracy at `alpha=8`.
- `last_prompt_token` was real but not dominant. It improved over baseline and
  over `assistant_only`, but it did not match `prompt_only` or `all_positions`.

## Mechanistic Read

- The earlier 3.3 failure of the stricter `assistant_only` harness was not
  evidence that the vectors were not steering 3.3.
- The steering effect appears to rely heavily on prompt-side intervention, not
  just generation-side injection.
- The likely next question is not "does prompt-side steering matter?" but
  "how much of the prompt tail is enough?"
