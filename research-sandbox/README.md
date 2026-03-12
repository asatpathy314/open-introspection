# Research Sandbox

This directory contains one-off experiment scripts and result folders that are
kept separate from the tracked steering evaluation harness.

## Folders

### `2026-03-12-position-scope/`
- Question: does open-ended steering depend on *where* the vector is injected?
- Script: `run_position_scope.py`
- Output: `results/position_scope_openended.jsonl`
- Conditions:
  - `assistant_only`
  - `prompt_only`
  - `last_prompt_token`
  - `all_positions`

### `2026-03-12-prompt-tail-window/`
- Question: if prompt-side steering matters, how many final prompt tokens are
  needed?
- Script: `run_prompt_tail_window.py`
- Output: `results/prompt_tail_window.jsonl`
- Conditions:
  - `baseline`
  - `assistant_only`
  - `tail_1`
  - `tail_2`
  - `tail_4`
  - `tail_8`
  - `prompt_all`
  - `all_positions`

### `2026-03-12-last-token-alpha-sweep/`
- Question: for `last_prompt_token`, at what alpha does judge-detectable
  steering appear?
- Script: `run_last_token_alpha_sweep.py`
- Output: `results/last_token_alpha_sweep.jsonl`
- Planned conditions:
  - `last_prompt_token`
  - `all_positions`
  - `alpha in {0, 2, 4, 8, 16}`

## Notes

- These scripts import shared utilities from `steering-eval/` but do not edit
  the tracked harness.
- Each script writes JSONL rows incrementally so interrupted runs can resume.
