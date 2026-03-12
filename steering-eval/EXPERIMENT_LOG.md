# Steering Eval Experiment Log

## Session 1 — 2026-03-11

### Decisions

1. **Model**: Using `meta-llama/Llama-3.1-70B-Instruct` (not 3.3) because 3.1 has a
   corresponding base model (`meta-llama/Llama-3.1-70B`) needed for base vs instruct
   comparison. Both confirmed online on NDIF.

2. **Existing vectors**: Only 3.3 vectors exist on disk. Must re-extract for 3.1.
   Extraction started as background task.

3. **Architecture**: Modular Python files in `steering-eval/`:
   - `config.py` — shared constants (model, paths, word lists, alpha sweeps)
   - `vectors.py` — loading + normalization (raw, unit, norm_matched)
   - `ndif_utils.py` — nnsight wrappers for steering injection
   - `level1_mcq.py` — Multiple-choice propensity (Tan et al.)
   - `level2_generation.py` — Open-ended generation (Rimsky et al.)
   - `level3_likelihood.py` — Likelihood-based (Pres et al.)
   - `stats.py` — Bootstrap CI, t-tests, BH correction, Spearman
   - `plots.py` — All 6+ figure types from the spec
   - `run_eval.py` — CLI orchestrator

4. **Spec compliance**: All three normalization variants, all alpha sweeps,
   data split (10 validation / 40 eval), resume support via JSONL, per-sample
   steerability analysis, anti-steerability fraction, BH FDR correction.

5. **LLM judge**: Using Claude Sonnet via Anthropic API for Level 2 identification
   and coherence judging.

### Critical: nnsight 0.6 constraints

- **Version**: Must use nnsight==0.6.0. Version 0.6.1+ has a server-side
  RecursionError with layer .output access on NDIF.
- **Source analysis**: nnsight 0.6 does source-code analysis of trace contexts.
  All proxy operations MUST be in the same file executed as `__main__`. Cross-module
  function calls containing `with model.trace()` fail with `WithBlockNotFoundError`.
- **Consequence**: Each eval level is a standalone script, not importable functions.
  The orchestrator (`run_eval.py`) dispatches via subprocess.
- **Shapes on NDIF**:
  - `model.model.layers[i].output[0]` -> `[seq_len, hidden_dim]` (no batch dim)
  - `model.lm_head.output` -> `[1, seq_len, vocab_size]`
- **Extraction**: `tracer.cache()` is broken. Must use explicit per-layer
  `.output[0][-1, :].save()`. Chunked into 4 traces of 20 layers each.

### Additional nnsight 0.6 constraint (discovered during extraction)

- **Dict/list subscript assignment fails silently**: `saves[i] = model.model.layers[i].output[0][-1, :].save()`
  inside trace contexts does NOT work — nnsight's source analysis drops the assignment.
  Must use explicit variable names: `h00 = ..., h01 = ..., h79 = ...`.

### Status
- [x] Code written for all 3 evaluation levels
- [x] nnsight 0.6 compatibility issues resolved
- [x] Vector extraction for 3.1 — 100 baseline words + 50 concepts, shape [80, 8192]
- [x] Level 1 MCQ sweep — 4500 trials (50 concepts × 3 layers × 6 alphas × 5 distractors)
- [ ] Level 2 generation eval (running — 20 concepts × 6 alphas × 5 gens at layer 40)
- [ ] Level 3 likelihood eval (running — 10 concepts at layer 40)
- [x] Report generation (preliminary with L1 + partial L2)
- [ ] Base model comparison

### Observations from existing 3.3 diagnostics
- Concept vectors have meaningful cosine structure (related pairs: mean 0.252, unrelated: -0.121 at layer 40)
- Baseline subtraction is effective (removes shared prompt component)
- Simple generation steering (diagnostics notebook) showed weak concept mention rates
  - Only "lightning" explicitly appeared in steered outputs at layer 40, alpha=8,16
  - This suggests MCQ/likelihood metrics may be more sensitive than string matching
- Concept vector norms are ~10-40% of baseline norms at middle layers
  - At alpha=4, perturbation is ~1-2x baseline norm
  - At alpha=16, perturbation is ~4-8x baseline norm (may hurt coherence)

## Session 2 — 2026-03-11 (continued)

### Level 1 MCQ Results (raw normalization, layers 20/40/60)
- **30/50 concepts** have positive steerability slope
- Mean steerability: 0.014 (positive = steering increases correct-concept logit)
- No concepts pass BH FDR correction at q=0.05 (power issue: only 5 distractors per concept)
- Top positively steerable: bread (0.203), volcanoes (0.154), sadness (0.147), dynasties (0.133)
- Top anti-steerable: information (-0.195), quarries (-0.153), youths (-0.134)
- Layer 40 and 60 show strongest effects in heatmap
- Abstract concepts (information, peace, memories) tend to anti-steer

### Level 2 Generation Results (partial, raw norm, layer 40)
- **Steered ID accuracy: 38%** vs baseline 7.7% (10% chance) — clear steering effect
- **Concept mention rate: 14%** steered vs 0% baseline
- **Coherence: 66%** steered vs 69% baseline — modest degradation
- At alpha=8+, model often starts response with the concept word directly
