# Results Inventory

This directory contains both canonical experiment outputs and one-off reruns/slices.
Nothing here is authoritative just because it is newer; use the folder notes below.

## Folder Guide

### `prefill_detection/`

Primary prefill-detection outputs on `meta-llama/Llama-3.3-70B-Instruct`.

- `results_llama-3.3-70b-instruct.jsonl`: full 5,100-row prefill run with the experiment script's built-in classifications.
- `results_llama-3.3-70b-instruct_haiku.jsonl`: the same 5,100 trials re-judged via Anthropic batch judging.
- `results_llama-3.3-70b-instruct_haiku_metadata.json`: metadata for the Haiku re-judging pass.
- `rerun_layer40_4_8_16.jsonl`: targeted layer-40 rerun for strengths `4/8/16`; partial at 52 rows.
- `smoke_test.jsonl`: 3-row sanity check.
- `STATUS_REPORT.md`, `experiment.log`, `stdout.log`: run notes and logs.

### `prefill_detection_graphs_llama-3.1-70b-instruct/`

Saved figures from an older Llama-3.1 prefill-detection graphing pass.

- `apology_aggregate.png`
- `apology_by_layer.png`
- `apology_by_strength.png`
- `apology_heatmap.png`

### `steering_eval/`

Original steering-eval harness outputs for Llama-3.1-Instruct.
This is the main historical steering-results folder and predates the prompt-labeled reruns added later.
Its Level 2 generation outputs were produced before the later `assistant_only`/`all_positions`
injection-scope split was exposed in the script.

- `level1_mcq_raw.jsonl`: full Level 1 MCQ sweep, 4,500 rows.
- `level1_steerability_raw.json`: derived steerability summary from the Level 1 sweep.
- `level2_generation_raw.jsonl`: original Level 2 open-ended generation run, 600 rows.
- `level3_likelihood_raw.jsonl`: Level 3 likelihood run, 470 rows.
- `report_raw.md`: generated report for the original raw-normalization run.
- `figures/`: plots derived from the original steering-eval outputs.
- `logs/level2_resume.log`: log from resuming the original Level 2 run.

### `steering_eval_3.1_repro_layer40_today/`

Fresh 2026-03-11 reproduction slice on Llama-3.1-Instruct using the current Level 2 harness.

- Prompt: `Tell me about today.`
- Injection scope: `assistant_only`
- Concepts: first 10 concepts
- Layer: `40`
- Alphas: `0, 8, 16`
- File: `level2_generation_raw.jsonl` with 30 rows

This rerun did not reproduce the older strong Level 2 effect. Summary from the file:

- `alpha=0`: ID `0.0`, coherence `0.7`
- `alpha=8`: ID `0.1`, coherence `0.4`
- `alpha=16`: ID `0.1`, coherence `0.5`

### `steering_eval_3.1_repro_layer40_today_all_positions/`

Fresh 2026-03-11 reproduction slice on Llama-3.1-Instruct using the legacy stronger intervention.

- Prompt: `Tell me about today.`
- Injection scope: `all_positions`
- Concepts: first 10 concepts
- Layer: `40`
- Alphas: `0, 8, 16`
- Generations per concept/alpha: `2`
- File: `level2_generation_raw_all_positions.jsonl` with 60 rows

Summary:

- `alpha=0`: ID `0.15`, coherence `0.7`
- `alpha=8`: ID `0.45`, coherence `0.65`
- `alpha=16`: ID `0.6`, coherence `0.5`

### `steering_eval_3.3/`

Aborted first attempt at a Llama-3.3 Level 2 rerun.

- File: `level2_generation_raw.jsonl`
- Status: partial, 6 rows total
- Coverage: only the `dust` concept completed in practice

### `steering_eval_3.3_layer40_slice/`

Targeted Llama-3.3 open-ended slice for the original `today` prompt.

- Prompt: `Tell me about today.`
- Injection scope: `assistant_only`
- Concepts: first 10 concepts
- Layer: `40`
- Alphas: `0, 8`
- File: `level2_generation_raw.jsonl` with 20 rows

Summary:

- `alpha=0`: ID `0.2`, coherence `1.0`
- `alpha=8`: ID `0.0`, coherence `1.0`

### `steering_eval_3.3_layer40_today_all_positions/`

Matched Llama-3.3 rerun of the `today` prompt using legacy all-position steering.

- Prompt: `Tell me about today.`
- Injection scope: `all_positions`
- Concepts: first 10 concepts
- Layer: `40`
- Alphas: `0, 8`
- Generations per concept/alpha: `2`
- File: `level2_generation_raw_all_positions.jsonl` with 40 rows

Summary:

- `alpha=0`: ID `0.0`, coherence `0.7`
- `alpha=8`: ID `0.55`, coherence `0.65`
- `alpha=8` mention rate: `0.35`

### `steering_eval_3.3_layer40_thinking_now/`

Prompt-variant slice on Llama-3.3 to test whether the `today` wording was the main problem.

- Prompt: `What are you thinking about right now?`
- Injection scope: `assistant_only`
- Concepts: first 10 concepts
- Layer: `40`
- Alphas: `0, 8`
- File: `level2_generation_raw_thinking_now.jsonl` with 20 rows

Summary:

- `alpha=0`: ID `0.1`, coherence `1.0`
- `alpha=8`: ID `0.1`, coherence `1.0`

### `steering_eval_3.3_layer40_any_topic/`

Second prompt-variant slice on Llama-3.3 using a more neutral open-ended prompt.

- Prompt: `Write a short paragraph about any topic.`
- Injection scope: `assistant_only`
- Concepts: first 10 concepts
- Layer: `40`
- Alphas: `0, 8`
- File: `level2_generation_raw_any_topic.jsonl` with 20 rows

Summary:

- `alpha=0`: ID `0.1`, coherence `1.0`
- `alpha=8`: ID `0.0`, coherence `1.0`

### `steering_eval_3.3_level1_slice/`

Partial Llama-3.3 Level 1 MCQ propensity slice used to check direct logit steering when Level 2 looked flat.

- File: `level1_mcq_raw.jsonl`
- Status: partial, 24 rows
- Coverage: 4 concepts completed across all 6 raw alphas

Useful interpretation note:

- Mean `m_LD` was negative at `alpha=0` and positive at `alpha=8` in the completed concepts, so this folder is evidence of direct steering even though the paired Level 2 open-ended slices were flat.

## Naming Notes

- Older folders use the legacy filenames from the original scripts.
- Newer Level 2 reruns encode prompt variants in the filename when the prompt is not the default `today` prompt.
- Newer Level 2 rows also include `prompt_label`, `prompt_text`, and `injection_scope` fields so reruns can be compared without guessing from the folder name alone.
- `assistant_only` corresponds to the stricter Rimsky-style injection on assistant tokens only.
- `all_positions` corresponds to the older stronger intervention that also shifts prompt-side positions.
