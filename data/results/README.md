# Results Inventory

This directory contains experiment outputs. Raw data files (JSONL, logs, figures)
are gitignored; only this README is tracked. Regenerate results by re-running the
experiment scripts.

## Directory Structure

```
data/results/
  prefill-detection/           # Prefill detection experiment (Llama-3.3-70B-Instruct)
    figures/                   # Saved plots (from an older Llama-3.1 graphing pass)
  steering-eval/               # Steering evaluation harness
    figures/                   # Plots from the main run
    logs/                      # Run logs
    3.1-layer40-today/         # Llama-3.1 repro, assistant_only scope
    3.1-layer40-today-all-positions/  # Llama-3.1 repro, all_positions scope
    3.3-partial/               # Aborted first Llama-3.3 attempt (6 rows)
    3.3-layer40-slice/         # Llama-3.3 "today" prompt, assistant_only
    3.3-layer40-today-all-positions/  # Llama-3.3 "today" prompt, all_positions
    3.3-layer40-thinking-now/  # Llama-3.3 "thinking about right now?" variant
    3.3-layer40-any-topic/     # Llama-3.3 "any topic" variant
    3.3-level1-slice/          # Llama-3.3 Level 1 MCQ propensity slice
```

## Prefill Detection

Primary outputs on `meta-llama/Llama-3.3-70B-Instruct`:

| File | Description |
|------|-------------|
| `results_llama-3.3-70b-instruct.jsonl` | Full 5,100-row prefill run with built-in classifications |
| `results_llama-3.3-70b-instruct_haiku.jsonl` | Same 5,100 trials re-judged via Claude Haiku batch |
| `results_llama-3.3-70b-instruct_haiku_metadata.json` | Metadata for the Haiku re-judging pass |
| `rerun_layer40_4_8_16.jsonl` | Targeted layer-40 rerun (52 rows) |
| `smoke_test.jsonl` | 3-row sanity check |
| `STATUS_REPORT.md` | Run notes |

## Steering Eval (main run)

Original harness outputs for Llama-3.1-Instruct (predates the injection-scope split):

| File | Rows | Description |
|------|------|-------------|
| `level1_mcq_raw.jsonl` | 4,500 | Full Level 1 MCQ sweep |
| `level1_steerability_raw.json` | — | Derived steerability summary |
| `level2_generation_raw.jsonl` | 600 | Level 2 open-ended generation |
| `level3_likelihood_raw.jsonl` | 470 | Level 3 likelihood |
| `report_raw.md` | — | Generated report |

## Steering Eval Variants

All variant runs use layer 40, first 10 concepts, alphas 0/8 (or 0/8/16 for 3.1).

| Directory | Model | Prompt | Scope | Key finding |
|-----------|-------|--------|-------|-------------|
| `3.1-layer40-today/` | 3.1 | "Tell me about today." | assistant_only | Weak effect (ID 0.1 at alpha=16) |
| `3.1-layer40-today-all-positions/` | 3.1 | "Tell me about today." | all_positions | Stronger (ID 0.6 at alpha=16) |
| `3.3-partial/` | 3.3 | default | assistant_only | Aborted, 6 rows |
| `3.3-layer40-slice/` | 3.3 | "Tell me about today." | assistant_only | Flat (ID 0.0 at alpha=8) |
| `3.3-layer40-today-all-positions/` | 3.3 | "Tell me about today." | all_positions | Effect (ID 0.55 at alpha=8) |
| `3.3-layer40-thinking-now/` | 3.3 | "What are you thinking about right now?" | assistant_only | Flat |
| `3.3-layer40-any-topic/` | 3.3 | "Write a short paragraph about any topic." | assistant_only | Flat |
| `3.3-level1-slice/` | 3.3 | MCQ | — | Positive m_LD shift at alpha=8 |

## Notes

- `assistant_only`: Rimsky-style injection on assistant tokens only.
- `all_positions`: Older stronger intervention that also shifts prompt-side positions.
- Newer Level 2 rows include `prompt_label`, `prompt_text`, and `injection_scope` fields.
