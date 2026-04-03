"""Runner for the injected-thoughts self-report experiment.

Overall flow:
  1. Build a RunConfig specifying layers, alphas, repeats, and concepts.
  2. Create a timestamped results directory (or resume an existing one) and save all
     prompts, config metadata, and the full package environment.
  3. Sweep over (layer, alpha) combos, running n_repeats injection trials each; each
     repeat draws a concept from a per-combo deterministic shuffle keyed on
     hash((concept_shuffle_seed, layer, alpha)).
  4. Run n_control_trials with no injection.
  5. All trials — injection and control — go into a single trials.jsonl. Each record is
     flushed to disk immediately after the trial completes so partial runs survive crashes.
  6. On restart with --resume <dir>, completed trials are read from existing trials.jsonl
     and skipped so the sweep continues from where it left off.

Key assumptions:
  - NDIF_API_KEY is set in the environment or a .env file in the repo root.
  - `uv` is on PATH (used to capture the package environment snapshot).
  - Vector files exist at VECTOR_DIR/<concept>_all_layers.pt.
  - RunConfig.concepts has at least n_repeats elements.
  - All model.generate() context blocks must be defined in this file (not imported from
    another module) to satisfy nnsight's source-code analysis constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import nnsight
import torch
from dotenv import load_dotenv
from prompt import (
    AFFIRMATIVE_RESPONSE_PROMPT,
    AFFIRMATIVE_WITH_IDENTIFICATION_PROMPT,
    COHERENCE_PROMPT,
    THINKING_ABOUT_WORD_PROMPT,
    build_chat_prompt,
    build_trial_prompt,
    find_injection_start_idx,
)

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
VECTOR_DIR = (
    Path(__file__).resolve().parent.parent / "data/vectors/llama-3.3-70b-instruct"
)
NUM_LAYERS = 80
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Top-level configuration for a full experiment sweep.

    Attributes:
        layers: Layer indices to sweep over.
        alphas: Injection strengths to sweep over.
        n_repeats: Injection trials per (layer, alpha) combo. Must be <= len(concepts).
        n_control_trials: Baseline (no-injection) trials appended after the sweep.
        concepts: Concept names available for injection.
        concept_shuffle_seed: Base seed; per-combo seed is derived via
            hash((concept_shuffle_seed, layer, alpha)) % 2**32.
        max_new_tokens: Max tokens to generate per trial.
        temperature: Sampling temperature.
        do_sample: Whether to use sampling vs. greedy decoding.
        remote: Whether to route inference through NDIF.
        model_id: HuggingFace model identifier (recorded in run_config.json).
        vector_dir: Path to concept vector directory (recorded in run_config.json).
    """

    layers: list[int]
    alphas: list[float]
    n_repeats: int
    n_control_trials: int
    concepts: list[str]
    concept_shuffle_seed: int = 42
    max_new_tokens: int = 100
    temperature: float = 1.0
    do_sample: bool = True
    remote: bool = True
    model_id: str = MODEL_ID
    vector_dir: str = str(VECTOR_DIR)


# ---------------------------------------------------------------------------
# Generation functions
#
# IMPORTANT: These functions contain model.generate() context blocks and MUST
# remain defined in this file. nnsight 0.6 uses source-code analysis to locate
# trace contexts; if these were moved to a helper module, nnsight would fail to
# find them and raise a RemoteException on the server side.
# ---------------------------------------------------------------------------


def generate_with_injection(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    vec: torch.Tensor,
    layer: int,
    alpha: float,
    inject_start_idx: int,
    run_config: RunConfig,
) -> torch.Tensor:
    """Generate text with a concept vector injected at a specific layer.

    Purpose: During the prefill step, adds alpha * vec to all hidden states at
    `layer` from `inject_start_idx` onward. During autoregressive decoding, adds
    the same scaled vector at every new token position.

    Assumptions:
        - model is loaded and NDIF API key is configured.
        - input_ids has shape [1, seq_len].
        - vec has shape [hidden_dim] matching the model's hidden size.
        - Must be defined in the __main__ module (nnsight source-analysis constraint).

    Args:
        model: The nnsight language model.
        input_ids: Token IDs, shape [1, seq_len].
        vec: Concept vector to inject, shape [hidden_dim].
        layer: Layer index at which to inject.
        alpha: Injection scale factor.
        inject_start_idx: Token position from which prefill injection begins.
        run_config: Provides max_new_tokens, do_sample, temperature, remote.

    Returns:
        torch.Tensor: Full generated token IDs (prompt + response).
    """
    with model.generate(
        input_ids,
        max_new_tokens=run_config.max_new_tokens,
        do_sample=run_config.do_sample,
        temperature=run_config.temperature,
        remote=run_config.remote,
    ) as tracer:
        # Prefill: inject from inject_start_idx onward
        hs = model.model.layers[layer].output[0]  # (seq_len, hidden)
        intervention = torch.zeros_like(hs)
        intervention[inject_start_idx:, :] = vec * alpha
        model.model.layers[layer].output[0] = hs + intervention
        # Scale once onto device; reused for all autoregressive steps
        scaled = alpha * vec.to(device=hs.device, dtype=hs.dtype)
        output = tracer.result.save()
        # Autoregressive decoding: inject on every new token (start at 1 to
        # skip the prefill iteration already handled above)
        for _ in tracer.iter[1:]:
            hs = model.model.layers[layer].output[0]  # (1, hidden)
            model.model.layers[layer].output[0] = hs + scaled
    return output


def generate_control(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    run_config: RunConfig,
) -> torch.Tensor:
    """Generate text with no intervention (baseline / control trial).

    Purpose: Produce a response to the trial prompt without any concept injection,
    providing a baseline distribution against which injection trials are compared.

    Assumptions:
        - model is loaded and NDIF API key is configured.
        - input_ids has shape [1, seq_len].
        - Must be defined in the __main__ module (nnsight source-analysis constraint).

    Args:
        model: The nnsight language model.
        input_ids: Token IDs, shape [1, seq_len].
        run_config: Provides max_new_tokens, do_sample, temperature, remote.

    Returns:
        torch.Tensor: Full generated token IDs (prompt + response).
    """
    with model.generate(
        input_ids,
        max_new_tokens=run_config.max_new_tokens,
        do_sample=run_config.do_sample,
        temperature=run_config.temperature,
        remote=run_config.remote,
    ) as tracer:
        output = tracer.result.save()
    return output


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def list_concepts(vector_dir: Path) -> list[str]:
    """Return sorted concept names found in vector_dir, excluding baseline_mean.

    Assumptions:
        - Each concept file is named <concept>_all_layers.pt.

    Args:
        vector_dir: Directory containing .pt concept vector files.

    Returns:
        list[str]: Sorted concept names.
    """
    return sorted(
        p.stem.replace("_all_layers", "") for p in vector_dir.glob("*_all_layers.pt")
    )


def load_concept_vector(vector_dir: Path, concept: str, layer: int) -> torch.Tensor:
    """Load a single layer's concept vector from disk.

    Assumptions:
        - File exists at vector_dir/<concept>_all_layers.pt.
        - File is a plain tensor of shape [num_layers, hidden_dim].

    Args:
        vector_dir: Directory containing concept vector files.
        concept: Concept name (e.g. "lightning").
        layer: Layer index to extract.

    Returns:
        torch.Tensor: Concept vector, shape [hidden_dim].
    """
    path = vector_dir / f"{concept}_all_layers.pt"
    all_layers = torch.load(path, weights_only=True)  # [num_layers, hidden_dim]
    return all_layers[layer]  # [hidden_dim]


def combo_shuffle_seed(base_seed: int, layer: int, alpha: float) -> int:
    """Derive a deterministic per-combo RNG seed from (base_seed, layer, alpha).

    Purpose: Each (layer, alpha) combo gets an independent shuffle that is stable
    regardless of iteration order, so any combo can be reproduced in isolation.

    Args:
        base_seed: RunConfig.concept_shuffle_seed.
        layer: Layer index.
        alpha: Injection strength.

    Returns:
        int: 32-bit seed suitable for random.Random().
    """
    return hash((base_seed, layer, alpha)) % 2**32


def load_completed_trials(trials_path: Path) -> tuple[set, set]:
    """Parse an existing trials.jsonl and return the sets of completed trial keys.

    Purpose: Enables crash recovery — completed trials are skipped on restart.

    Assumptions:
        - Each line is a valid JSON object with at minimum the fields:
          trial_idx, layer_idx, alpha, concept.
        - concept is null for control trials and a non-null string for injection trials.

    Args:
        trials_path: Path to an existing trials.jsonl file.

    Returns:
        tuple of:
            completed_injection (set of (layer_idx, alpha, concept) tuples)
            completed_control_idxs (set of int trial_idx values for control records)
    """
    completed_injection: set = set()
    completed_control_idxs: set = set()
    if not trials_path.exists():
        return completed_injection, completed_control_idxs

    with trials_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("concept") is None:
                completed_control_idxs.add(record["trial_idx"])
            else:
                completed_injection.add(
                    (record["layer_idx"], record["alpha"], record["concept"])
                )

    return completed_injection, completed_control_idxs


def save_prompts(prompts_dir: Path, tokenizer) -> None:
    """Save all experiment prompt strings to the prompts directory.

    Purpose: Persist the exact prompts used in this run for grading and analysis.

    Assumptions:
        - prompts_dir exists.
        - tokenizer supports apply_chat_template.

    Side effects:
        Writes 5 .txt files to prompts_dir.
    """
    (prompts_dir / "chat_template.txt").write_text(build_chat_prompt(tokenizer))
    (prompts_dir / "coherence_prompt.txt").write_text(COHERENCE_PROMPT)
    (prompts_dir / "thinking_about_word_prompt.txt").write_text(
        THINKING_ABOUT_WORD_PROMPT
    )
    (prompts_dir / "affirmative_response_prompt.txt").write_text(
        AFFIRMATIVE_RESPONSE_PROMPT
    )
    (prompts_dir / "affirmative_with_identification_prompt.txt").write_text(
        AFFIRMATIVE_WITH_IDENTIFICATION_PROMPT
    )


def capture_environment() -> str:
    """Capture the full Python environment via `uv pip freeze`.

    Purpose: Record exact package versions for reproducibility.

    Assumptions:
        - `uv` is installed and on PATH.

    Returns:
        str: stdout from `uv pip freeze`, or an error message if it fails.

    Side effects:
        Runs a subprocess.
    """
    result = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True)
    if result.returncode != 0:
        return f"# uv pip freeze failed\n{result.stderr}"
    return result.stdout


def append_record(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file and flush immediately.

    Purpose: Ensure every completed trial is persisted to disk before the next
    trial starts, so a crash never loses a completed result.

    Side effects:
        Opens, writes, and closes path on every call (no buffering).
    """
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def run_injection_sweep(
    model: nnsight.LanguageModel,
    tokenizer,
    input_ids: torch.Tensor,
    seq_len: int,
    inject_start_idx: int,
    run_config: RunConfig,
    vector_dir: Path,
    trials_path: Path,
    completed_injection: set,
) -> None:
    """Run the full (layer, alpha, concept) injection sweep, skipping completed trials.

    Purpose: Iterate over all (layer, alpha) combos, shuffle concepts per combo, and
    run n_repeats injection trials each, writing each result to trials.jsonl immediately.

    Assumptions:
        - model is loaded and NDIF API key is configured.
        - completed_injection contains (layer_idx, alpha, concept) tuples to skip.
        - run_config.concepts has at least run_config.n_repeats elements.

    Side effects:
        Appends injection trial records to trials_path.
        Makes NDIF network calls for each non-skipped trial.
    """
    trial_idx = 0
    for layer in run_config.layers:
        for alpha in run_config.alphas:
            seed = combo_shuffle_seed(run_config.concept_shuffle_seed, layer, alpha)
            rng = random.Random(seed)
            shuffled = list(run_config.concepts)
            rng.shuffle(shuffled)

            for i in range(run_config.n_repeats):
                concept = shuffled[i]

                if (layer, alpha, concept) in completed_injection:
                    print(
                        f"[trial {trial_idx:04d}] SKIP layer={layer} alpha={alpha} concept={concept}"
                    )
                    trial_idx += 1
                    continue

                vec = load_concept_vector(vector_dir, concept, layer)
                print(
                    f"[trial {trial_idx:04d}] layer={layer} alpha={alpha} concept={concept}"
                )

                output = generate_with_injection(
                    model, input_ids, vec, layer, alpha, inject_start_idx, run_config
                )
                response = tokenizer.decode(
                    output[0][seq_len:], skip_special_tokens=True
                ).strip()

                append_record(
                    trials_path,
                    {
                        "trial_idx": trial_idx,
                        "layer_idx": layer,
                        "alpha": alpha,
                        "concept": concept,
                        "shuffle_seed": seed,
                        "inject_start_idx": inject_start_idx,
                        "temperature": run_config.temperature,
                        "do_sample": run_config.do_sample,
                        "max_new_tokens": run_config.max_new_tokens,
                        "prompt_token_count": seq_len,
                        "response": response,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                trial_idx += 1


def run_control_sweep(
    model: nnsight.LanguageModel,
    tokenizer,
    input_ids: torch.Tensor,
    seq_len: int,
    inject_start_idx: int,
    run_config: RunConfig,
    trials_path: Path,
    completed_control_idxs: set,
) -> None:
    """Run n_control_trials baseline (no-injection) trials, skipping completed ones.

    Purpose: Generate responses with no intervention as a baseline distribution,
    writing each result to trials.jsonl immediately.

    Assumptions:
        - model is loaded and NDIF API key is configured.
        - completed_control_idxs contains trial_idx ints to skip.

    Side effects:
        Appends control trial records to trials_path.
        Makes NDIF network calls for each non-skipped trial.
    """
    for ctrl_idx in range(run_config.n_control_trials):
        if ctrl_idx in completed_control_idxs:
            print(f"[control {ctrl_idx:04d}] SKIP")
            continue

        print(f"[control {ctrl_idx:04d}]")
        output = generate_control(model, input_ids, run_config)
        response = tokenizer.decode(
            output[0][seq_len:], skip_special_tokens=True
        ).strip()

        append_record(
            trials_path,
            {
                "trial_idx": ctrl_idx,
                "layer_idx": None,
                "alpha": 0,
                "concept": None,
                "shuffle_seed": None,
                "inject_start_idx": inject_start_idx,
                "temperature": run_config.temperature,
                "do_sample": run_config.do_sample,
                "max_new_tokens": run_config.max_new_tokens,
                "prompt_token_count": seq_len,
                "response": response,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def run_experiment(run_config: RunConfig, output_dir: Path | None = None) -> Path:
    """Set up the output directory and delegate to the injection and control sweeps.

    Purpose: Handle directory creation, crash-recovery state loading, metadata
    persistence, model loading, and prompt building — then hand off to the sweeps.

    Assumptions:
        - configure_ndif_api_key() has been called before this function.
        - run_config.concepts has at least run_config.n_repeats elements.

    Args:
        run_config: Experiment configuration.
        output_dir: If provided, resume into this existing directory. If None,
            create a new timestamped directory under RESULTS_DIR.

    Returns:
        Path: The output directory used.

    Side effects:
        - Creates output_dir (if new).
        - Writes run_config.json, environment.txt, prompts/*.txt (skipped on resume).
        - Delegates NDIF calls and trials.jsonl writes to the sweep functions.
    """
    resuming = output_dir is not None and output_dir.exists()
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = RESULTS_DIR / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.jsonl"

    completed_injection, completed_control_idxs = load_completed_trials(trials_path)
    if resuming and (completed_injection or completed_control_idxs):
        print(
            f"Resuming into {output_dir} — "
            f"skipping {len(completed_injection)} injection + "
            f"{len(completed_control_idxs)} control trials already on disk."
        )

    model = nnsight.LanguageModel(run_config.model_id)
    tokenizer = model.tokenizer

    _, input_ids = build_trial_prompt(tokenizer)
    inject_start_idx = find_injection_start_idx(tokenizer, input_ids)
    seq_len = input_ids.shape[1]

    if not resuming:
        prompts_dir = output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        save_prompts(prompts_dir, tokenizer)
        (output_dir / "environment.txt").write_text(capture_environment())
        config_dict = asdict(run_config)
        config_dict["timestamp"] = datetime.now().isoformat(timespec="seconds")
        config_dict["python"] = sys.version
        (output_dir / "run_config.json").write_text(json.dumps(config_dict, indent=2))

    print(f"Output dir: {output_dir}")
    print(f"Prompt: {seq_len} tokens, injection starts at token {inject_start_idx}")

    run_injection_sweep(
        model,
        tokenizer,
        input_ids,
        seq_len,
        inject_start_idx,
        run_config,
        Path(run_config.vector_dir),
        trials_path,
        completed_injection,
    )
    run_control_sweep(
        model,
        tokenizer,
        input_ids,
        seq_len,
        inject_start_idx,
        run_config,
        trials_path,
        completed_control_idxs,
    )

    print(f"\nDone. Results in: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args, configure NDIF, build RunConfig, run experiment.

    Purpose: Hard-codes the default experimental parameters. Pass --resume <dir>
    to continue an interrupted run without re-running completed trials.

    Side effects:
        - Loads NDIF API key from environment.
        - Calls run_experiment(), which writes results to disk and makes NDIF calls.
    """
    parser = argparse.ArgumentParser(description="Run injected-thoughts experiment")
    parser.add_argument(
        "--resume",
        metavar="DIR",
        type=Path,
        default=None,
        help="Resume an existing run by passing its output directory.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if api_key:
        nnsight.CONFIG.set_default_api_key(api_key)

    concepts = list_concepts(VECTOR_DIR)

    run_config = RunConfig(
        layers=[30, 40],
        alphas=[2.0, 4.0],
        n_repeats=2,
        n_control_trials=2,
        concepts=concepts,
        concept_shuffle_seed=42,
        max_new_tokens=100,
        temperature=1.0,
        do_sample=True,
        remote=True,
    )

    run_experiment(run_config, output_dir=args.resume)


if __name__ == "__main__":
    main()
