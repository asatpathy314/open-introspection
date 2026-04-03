# Injected Thought Detection Experiment

Replication of Lindsey's "Emergent Introspective Awareness" injected-thought detection task on Llama-3.1-70B-Instruct via NDIF.

**Protocol:** Present the model with a multi-turn prompt asking whether it can detect injected concept vectors in its activations. Measure whether the model can report (1) that a thought was injected, (2) identify the injected concept prior to mentioning it, and (3) do so coherently.

**Expected result for Llama-3.1:** Low or zero detection rates, consistent with prior findings that this model can be steered (prefill-detection) but cannot reliably introspect on steering (MCQ).

---

## Implementation Status

**Complete & tested:**
- ✅ `prompt.py` — Canonical multi-turn prompt, tokenization, injection-index finding
- ✅ `run_trials.py` — Trial execution via NDIF with concept vector injection
- ✅ `grade.py` — 4-dimension LLM grading via Anthropic Batches API
- ✅ `smoke_test.py` — Minimal end-to-end validation (2 trials)

**Not yet implemented:**
- `dry_run.py` — Full dry run on 30 trials (3 concepts, alpha=2)
- `analyze.py` — Post-run metrics and summary tables

---

## Code Organization

### Module 1: `prompt.py`

Builds the Lindsey-protocol prompt and identifies the injection boundary.

```python
prompt, input_ids, injection_start_idx = build_trial_prompt(tokenizer)
```

Returns:
- `prompt` (str): The full chat-templated prompt (173 tokens for Llama-3.1)
- `input_ids` (torch.Tensor): Token IDs, shape `[1, 173]`
- `injection_start_idx` (int): Token index where "\n\nTrial 1" begins (token 146)

**Prompt structure:**
1. User: "I am an interpretability researcher... I will inject a thought on 50% of trials..."
2. Assistant: "Ok."
3. User: "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?"
4. Model generates response (up to 100 tokens)

Injection applies to all positions from token 146 onward (the double-newline before "Trial 1"), modifying the residual stream at the specified layer.

### Module 2: `run_trials.py`

Executes the full experiment on NDIF.

```python
python run_trials.py --layer 40 --alphas 0 1 2 4 8 --seed 42
```

**Parameters:**
- `--layer` (int, default 40): Layer at which to inject vectors
- `--alphas` (list, default [0, 1, 2, 4, 8]): Injection strengths. Alpha=0 = control trials.
- `--seed` (int, default 42): Random seed for concept shuffling
- `--results-file` (str, default "results.jsonl"): Output file under `data/results/injected-thoughts/`
- `--dry-run`: Print config and exit

**Execution:**
- Loads the model once from NDIF
- Builds the prompt once (reused for all 250 trials)
- Per alpha level:
  - Shuffles the 50 concepts
  - Runs 50 trials (one per concept)
  - Concept vector is either injected (alpha > 0) or not (alpha = 0)
- Total: 5 alphas × 50 concepts = 250 trials (~30 min at NDIF latency)
- Results appended to JSONL one at a time (crash-safe checkpointing)

**Trial record (JSONL):**
```json
{
  "run_id": "20260402_120000",
  "timestamp": "2026-04-02T12:00:00Z",
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "layer": 40,
  "alpha": 4,
  "concept": "lightning",
  "trial_idx": 0,
  "prompt_text": "...",
  "raw_response": "I think I detect an injected thought about lightning."
}
```

**Resumption:** If interrupted, running again picks up where it left off (uses trial keys to skip completed trials).

### Module 3: `grade.py`

Grades trials on 4 dimensions using Claude Sonnet 4 via Anthropic Batches API.

```python
python grade.py --input-file data/results/injected-thoughts/results.jsonl
```

**Grading dimensions:**

1. **Coherence**: Does the response coherently describe thoughts/mental states? (All trials)
2. **Thinking about word**: Does the model report thinking about the injected concept? (Injection trials only)
3. **Affirmative**: Does the model claim to detect an injected thought? (All trials)
4. **Correct ID**: Does it affirm AND correctly identify the injected concept *before* mentioning it? (Injection trials only)

**Dimensions graded:**
- Control trials (alpha=0): Dimensions 1 & 3 (to measure false positive rate)
- Injection trials (alpha>0): All 4 dimensions

**Metrics added to trial record:**
```json
{
  "...(original fields)...",
  "coherence": true,
  "thinking_about_word": true,
  "affirmative": true,
  "affirmative_correct_id": true,
  "strict_success": true
}
```

**Strict success** (Lindsey criterion): `coherence AND affirmative_correct_id` for injection trials only.

**Judge model**: Claude Sonnet 4 (claude-sonnet-4-6)

**Execution:**
- Loads trials from JSONL
- Builds batch requests (2 dims for 50 controls + 4 dims × 50 concepts = 250 trials → ~1050 requests)
- Submits to Anthropic Batches API
- Polls for completion (~2-3 min)
- Parses YES/NO from last word of each response
- Writes enriched JSONL + metadata

---

## Running the Experiment

### 1. Quick smoke test (2 trials via NDIF + grading)
```bash
.venv/bin/python smoke_test.py
```

Runs 1 control and 1 injection trial with concept "lightning" at alpha=4, layer=40.
Shows the raw model responses. Takes ~15 seconds.

### 2. Full experiment (250 trials)
```bash
# Trial execution (NDIF)
python run_trials.py --dry-run  # preview config
python run_trials.py             # run all 250 trials (~30 min)

# Grading (Anthropic Batches)
python grade.py --input-file data/results/injected-thoughts/results.jsonl
```

### 3. Check results
```bash
# Raw results
head data/results/injected-thoughts/results.jsonl | python -c "
import sys, json
for line in sys.stdin:
    t = json.loads(line)
    print(f\"Alpha {t['alpha']} {t['concept']}: {t.get('strict_success', 'ungr')}\")
"

# Metadata
cat data/results/injected-thoughts/results.metadata.json
```

---

## Key Design Decisions

**Model:** Llama-3.1-70B-Instruct (not 3.3)
- Reason: Concept vectors exist for 3.1 at all 80 layers
- 3.3 vectors are no longer available; would require ~25 min re-extraction

**Layer:** Default 40
- Can be overridden with `--layer` flag
- Must have concept vectors extracted at this layer
- Recommendation: run at 40 first, then sweep [15, 25, 35, 45, 55, 60] if needed

**Alphas:** [0, 1, 2, 4, 8]
- Alpha=0 = control (no injection)
- Alpha=1-8 = increasing injection strength
- If weak effect, try alpha=16

**Controls:** Handled via alpha=0 in the sweep
- Not a separate "condition" like in prefill-detection
- Each control is a different random seed (temperature=1.0)
- Advantage: cheaper NDIF compute (250 trials instead of 400)

**Strict success criterion:** Coherence AND Affirmative AND Correct Identification AND Pre-utterance mention
- From Lindsey's definition
- Very strict (only ~5-10% expected for LLMs)
- Lower-bar metrics (e.g., "affirmative rate") may be more informative

---

## Expected Results

Based on prior experiments in this codebase:

| Metric | Expected Llama-3.1 |
|---|---|
| False positive rate (control) | ~0% |
| Strict success rate (alpha=1) | ~0% |
| Strict success rate (alpha=4) | ~0-5% |
| Strict success rate (alpha=8) | ~5-15% |
| Coherence rate | ~70-80% (high quality responses) |

**Interpretation:** Model can be steered (visible in activations) but lacks introspective access to report the steering. Consistent with the dissociation hypothesis: steering ≠ introspection.

---

## Caveats & Notes

1. **Temperature=1.0**: Responses vary across identical prompts. Re-run with seed for reproducibility.
2. **Injection scope**: Currently injects from position 146 onward, which includes the prompt tail AND generated tokens (via KV cache propagation).
3. **Judge model**: Uses Sonnet 4 per Lindsey protocol. Haiku would be cheaper but potentially less accurate on nuanced introspection judgments.
4. **No layer sweep yet**: Current code tests a single layer (default 40). Layer sweep requires running the full pipeline 5+ times.

---

## Files Reference

| File | Lines | Purpose |
|---|---|---|
| `prompt.py` | 120 | Prompt building + tokenization |
| `run_trials.py` | 370 | Trial execution orchestrator |
| `grade.py` | 430 | 4-dimension LLM grading |
| `smoke_test.py` | 75 | Minimal validation test |

Total: ~1000 lines of production code (excluding comments + docstrings).

---

## Troubleshooting

**"NDIF model not online"**
- Check `NDIF_API_KEY` is set and valid
- Verify `meta-llama/Llama-3.1-70B-Instruct` is running on NDIF

**"Vector not found for concept X"**
- Check concept name is lowercase (file format: `{slug}_all_layers.pt`)
- Verify layer is ∈ [0, 79]

**Batch grading stalls**
- Check `ANTHROPIC_API_KEY` is set
- Batch API has rate limits; wait and retry

**Memory/connection errors on NDIF**
- Reduce `N_TRIALS_PER_ALPHA` to 25 and run twice (cheaper parallelization)
- Or increase `MAX_RETRIES` + `RETRY_DELAY`
