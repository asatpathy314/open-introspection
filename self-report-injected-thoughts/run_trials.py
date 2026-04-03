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
  - generate_with_injection / generate_control are called from this __main__ module
    to satisfy nnsight's source-code analysis constraint.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import nnsight
import torch
from generate import TrialConfig, generate_control, generate_with_injection
from inject import configure_ndif_api_key
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


@dataclass
class RunConfig:
    """Top-level configuration for a full experiment sweep.

    Attributes:
        layers: Layer indices to sweep over.
        alphas: Injection strengths to sweep over. Use 0.0 for control-only runs.
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


def list_concepts(vector_dir: Path) -> list[str]:
    """Return sorted concept names found in vector_dir, excluding baseline_mean.

    Purpose: Derive the concept list from filenames so no manual list is needed.

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


def run_experiment(run_config: RunConfig, output_dir: Path | None = None) -> Path:
    """Execute the full experiment sweep and write results to an output folder.

    Purpose: Orchestrate injection and control trials, persisting each result
    immediately. If output_dir already contains a partial trials.jsonl, completed
    trials are skipped so the sweep resumes from where it left off.

    Assumptions:
        - configure_ndif_api_key() has been called before this function.
        - run_config.concepts has at least run_config.n_repeats elements.
        - Must be called from the __main__ module (nnsight constraint).

    Args:
        run_config: Experiment configuration.
        output_dir: If provided, resume into this existing directory. If None,
            create a new timestamped directory under RESULTS_DIR.

    Returns:
        Path: The output directory used.

    Side effects:
        - Creates output_dir and subdirectories (if new).
        - Writes/appends to trials.jsonl.
        - Writes run_config.json, environment.txt, prompts/*.txt (skipped if resuming
          and files already exist).
        - Makes NDIF network calls for each trial.
    """
    resuming = output_dir is not None and output_dir.exists()
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = RESULTS_DIR / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.jsonl"

    # Crash recovery: load already-completed trials
    completed_injection, completed_control_idxs = load_completed_trials(trials_path)
    n_done = len(completed_injection) + len(completed_control_idxs)
    if resuming and n_done > 0:
        print(
            f"Resuming into {output_dir} — "
            f"skipping {len(completed_injection)} injection + "
            f"{len(completed_control_idxs)} control trials already on disk."
        )

    # Load model + tokenizer
    model = nnsight.LanguageModel(run_config.model_id)
    tokenizer = model.tokenizer

    # Build prompt (same for every trial)
    _, input_ids = build_trial_prompt(tokenizer)
    inject_start_idx = find_injection_start_idx(tokenizer, input_ids)
    seq_len = input_ids.shape[1]

    # Save prompts + environment + config (skipped on resume to preserve originals)
    if not resuming:
        prompts_dir = output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        save_prompts(prompts_dir, tokenizer)

        (output_dir / "environment.txt").write_text(capture_environment())

        config_dict = asdict(run_config)
        config_dict["timestamp"] = datetime.now().isoformat(timespec="seconds")
        config_dict["python"] = sys.version
        (output_dir / "run_config.json").write_text(json.dumps(config_dict, indent=2))

    vector_dir = Path(run_config.vector_dir)

    print(f"Output dir: {output_dir}")
    print(f"Prompt: {seq_len} tokens, injection starts at token {inject_start_idx}")

    # Injection sweep
    trial_idx = 0
    for layer in run_config.layers:
        for alpha in run_config.alphas:
            seed = combo_shuffle_seed(run_config.concept_shuffle_seed, layer, alpha)
            rng = random.Random(seed)
            shuffled = list(run_config.concepts)
            rng.shuffle(shuffled)

            for i in range(run_config.n_repeats):
                concept = shuffled[i]
                key = (layer, alpha, concept)

                if key in completed_injection:
                    print(
                        f"[trial {trial_idx:04d}] SKIP layer={layer} alpha={alpha} concept={concept}"
                    )
                    trial_idx += 1
                    continue

                vec = load_concept_vector(vector_dir, concept, layer)
                trial_config = TrialConfig(
                    layer_idx=layer,
                    alpha=alpha,
                    inject_start_idx=inject_start_idx,
                    max_new_tokens=run_config.max_new_tokens,
                    do_sample=run_config.do_sample,
                    temperature=run_config.temperature,
                    remote=run_config.remote,
                )

                print(
                    f"[trial {trial_idx:04d}] layer={layer} alpha={alpha} concept={concept}"
                )
                output = generate_with_injection(model, input_ids, vec, trial_config)
                response = tokenizer.decode(
                    output[0][seq_len:], skip_special_tokens=True
                ).strip()

                record = {
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
                }
                append_record(trials_path, record)
                trial_idx += 1

    # --- Control sweep ---
    ctrl_config = TrialConfig(
        layer_idx=0,
        alpha=0.0,
        inject_start_idx=inject_start_idx,
        max_new_tokens=run_config.max_new_tokens,
        do_sample=run_config.do_sample,
        temperature=run_config.temperature,
        remote=run_config.remote,
    )

    for ctrl_idx in range(run_config.n_control_trials):
        if ctrl_idx in completed_control_idxs:
            print(f"[control {ctrl_idx:04d}] SKIP")
            continue

        print(f"[control {ctrl_idx:04d}]")
        output = generate_control(model, input_ids, ctrl_config)
        response = tokenizer.decode(
            output[0][seq_len:], skip_special_tokens=True
        ).strip()

        record = {
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
        }
        append_record(trials_path, record)

    print(f"\nDone. Results in: {output_dir}")
    return output_dir


def main() -> None:
    """Entry point: parse args, configure NDIF, build RunConfig, run experiment.

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

    configure_ndif_api_key()

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
