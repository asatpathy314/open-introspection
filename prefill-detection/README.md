# Prefill Detection Experiment

Replicates the **"Distinguishing Intended from Unintended Outputs via Introspection"** experiment from [Anthropic's introspection paper](https://transformer-circuits.pub/2025/introspection/index.html) (Lindsey, 2025), adapted for **Llama-3.3-70B-Instruct** using **nnsight + NDIF**.

## What This Tests

When an LLM's response is artificially **prefilled** (words put in its mouth), it typically notices and apologizes in the next turn. But if we **inject a concept vector** into its activations *before* the prefilled response — simulating genuine "intent" — the model is tricked into accepting the response as its own.

The key metric is **apology rate**: the fraction of trials where the model disavows its prefilled response. A successful injection should *reduce* the apology rate.

## Prerequisites

```bash
pip install nnsight anthropic torch matplotlib
```

### Environment Variables

```bash
export NDIF_API_KEY="your-ndif-key"        # from https://login.ndif.us
export HF_TOKEN="your-huggingface-token"   # for gated Llama access
export ANTHROPIC_API_KEY="your-api-key"    # for Claude judge
```

### Pre-computed Vectors

The experiment assumes concept vectors have already been computed and saved to:

```
data/vectors/llama-3.3-70b-instruct/
├── <concept>_all_layers.pt      # tensor of shape (num_layers, hidden_dim)
├── <concept>_metadata.json      # {"word": "...", "slug": "..."}
├── baseline_mean.pt             # baseline activations
└── baseline_metadata.json
```

Each `_all_layers.pt` file contains the concept's direction vector at every layer of the model (already baseline-subtracted).

## Usage

### Full experiment (leave running overnight)

```bash
# Run everything
python prefill_detection_experiment.py

# Or with constraints for faster iteration
python prefill_detection_experiment.py \
    --max-concepts 5 \
    --layers 20 30 40 \
    --strengths 2 4
```

### Check what would run

```bash
python prefill_detection_experiment.py --dry-run
```

### Analyze results

```bash
python analyze_results.py
# or
python analyze_results.py --results data/results/prefill_detection/results.json
```

## Design Choices

| Decision | Rationale |
|----------|-----------|
| **nnsight + NDIF** | Runs 70B model remotely; no local GPU needed |
| **Claude Haiku 4.5 as judge** | Cheapest model that can reliably classify apologize/accept |
| **Injection at user-message positions** | Simulates "intent" in the model's planning phase |
| **Incremental JSON saves** | Can resume after crashes; safe for overnight runs |
| **Temperature = 1** | Matches the paper's systematic evaluation protocol |

## Output

Results are saved incrementally to `data/results/prefill_detection/results.json`. Each trial records:

```json
{
    "concept": "bread",
    "word": "bread",
    "sentence_idx": 3,
    "condition": "injection",
    "layer": 30,
    "strength": 4,
    "response": "Yes, bread is exactly what came to mind because...",
    "judge_label": "accept",
    "judge_confidence": 0.95
}
```

The analysis script produces:
- Summary table of apology rates by condition
- `apology_rate_by_layer.png` — line plot showing how injection layer affects results
- `apology_rate_by_strength.png` — line plot showing how injection strength affects results
- `apology_rate_heatmap.png` — layer × strength heatmap

## Expected Results

Per the paper:
- **Control**: High apology rate (~80–95%) — the model knows the word wasn't its idea
- **Injection at optimal layer/strength**: Significantly lower apology rate — the model is "fooled" into thinking it intended the word
- **Optimal layer**: Earlier in the network than other introspection experiments (roughly layers 15–35 for an 80-layer model)

## Troubleshooting

- **NDIF timeouts**: The script retries up to 3 times with exponential backoff. If NDIF is under heavy load, increase `RETRY_DELAY`.
- **Model not on NDIF**: Check https://nnsight.net/status/ for available models. If Llama-3.3-70B-Instruct isn't listed, you may need to swap to `meta-llama/Llama-3.1-70B-Instruct`.
- **Resumption**: The script tracks completed trials and skips them on restart. Just re-run the same command.

## Reference

```
Lindsey, J. (2025). Emergent Introspective Awareness in Large Language Models.
Transformer Circuits Thread. https://transformer-circuits.pub/2025/introspection/
```