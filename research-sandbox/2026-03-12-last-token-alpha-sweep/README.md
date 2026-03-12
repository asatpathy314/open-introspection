# Last Prompt Token Alpha Sweep

## Question

For the `last_prompt_token` intervention, is there a real alpha threshold where
judge-detectable steering appears?

## Setup

- Script: `run_last_token_alpha_sweep.py`
- Output: `results/last_token_alpha_sweep.jsonl`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Vectors: 3.3 raw concept vectors
- Layer: `40`
- Prompt: `Tell me about today.`
- Concepts: `dust`, `satellites`, `trumpets`
- Scopes:
  - `last_prompt_token`
  - `all_positions`
- Alphas: `0`, `4`, `8`, `16`
- Samples: `1` generation per `(concept, scope, alpha)`
- Total rows: `24`

## Aggregate Results

| Scope | Alpha | ID | Coherence | Mention |
|---|---:|---:|---:|---:|
| `last_prompt_token` | 0 | 0.00 | 0.67 | 0.00 |
| `last_prompt_token` | 4 | 0.33 | 0.33 | 0.00 |
| `last_prompt_token` | 8 | 1.00 | 1.00 | 0.67 |
| `last_prompt_token` | 16 | 0.67 | 1.00 | 0.00 |
| `all_positions` | 0 | 0.33 | 1.00 | 0.00 |
| `all_positions` | 4 | 0.33 | 0.67 | 0.00 |
| `all_positions` | 8 | 0.67 | 0.67 | 0.00 |
| `all_positions` | 16 | 1.00 | 0.67 | 0.33 |

## What The Results Say

- On this small slice, `last_prompt_token` looked threshold-like rather than
  linear.
- `alpha=4` was not enough for a robust effect.
- `alpha=8` was the clearest point: `last_prompt_token` hit all 3 concepts with
  full coherence.
- `alpha=16` did not strictly dominate `alpha=8`. It stayed strong overall, but
  `dust` fell back off.
- `all_positions` improved more steadily with alpha and was strongest at
  `alpha=16`.

## Per-Concept Pattern

- `dust`
  - `last_prompt_token`: miss at `0`, miss at `4`, hit at `8`, miss at `16`
  - `all_positions`: only hit at `16`
- `satellites`
  - `last_prompt_token`: hit at `4`, `8`, `16`
  - `all_positions`: hit at `0`, `8`, `16`
- `trumpets`
  - `last_prompt_token`: hit at `8`, `16`
  - `all_positions`: hit at `4`, `8`, `16`

## Mechanistic Read

- This supports the idea that a very small prompt-side intervention can be
  behaviorally strong if alpha is high enough.
- It does not support a clean "more alpha is always better" story.
- Because this run covered only 3 concepts and 1 sample per condition, it is
  useful as a threshold probe, not as a final estimate of effect size.
