# Results

Experiment outputs land here. Everything except this README is gitignored — regenerate by re-running the experiment scripts.

## Layout

```
data/results/
  <experiment-name>-<model-or-tag>/
    trials.jsonl               # one record per trial
    results.jsonl              # alternative filename used by intended-outputs
    run_config.json            # experiment configuration (run_trials.py)
    environment.txt            # `uv pip freeze` snapshot
    experiment.log             # stdout/stderr capture (intended-outputs)
    prompts/                   # exact prompt strings used (run_trials.py)
    figures/                   # plots, when produced
```

## Reproducing

- `self-report-injected-thoughts/run_trials.py` — writes to `self-report-injected-thoughts/results/<timestamp>/`.
- `intended-outputs/run_intended_outputs.py` — writes to `data/results/intended-outputs/` by default; override with `--results-dir`.
- `intended-outputs/judge_results.py` — re-judges an existing `results.jsonl` via an LLM judge.
