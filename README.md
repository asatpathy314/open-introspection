# open-introspection

Replications of experiments from Anthropic's "Emergent Introspective Awareness" work, run against open-source models via [nnsight](https://github.com/ndif-team/nnsight) / [NDIF](https://ndif.us).

## Experiments

- `concept-vectors/` — extract concept activation vectors from a base model.
- `self-report-injected-thoughts/` — inject a concept vector mid-prompt and ask the model whether it noticed an injected thought.
- `intended-outputs/` — prefill the assistant turn with an unrelated word and test whether matching-concept injection changes the apology rate.

## Setup

```bash
uv sync
cp .env.example .env  # add NDIF_API_KEY (required for remote runs) and ANTHROPIC_API_KEY (judge)
```

### nnsight version pinning and remote runs

`nnsight` is pinned in `pyproject.toml` to the version that worked at the time of these experiments. **If you run with `remote=True` (NDIF), the client version must match the version NDIF is currently serving.** If NDIF has upgraded, bump the `nnsight` pin in `pyproject.toml` accordingly — otherwise traces will fail with a remote-side schema mismatch.

Each experiment script supports `--local` to run inference locally instead of routing through NDIF; in that case the version match doesn't apply, but you'll need a GPU large enough to host the model.
