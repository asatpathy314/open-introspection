"""Injected-thought detection experiment runner.

Runs injection and control trials via NDIF, following Lindsey's protocol.
Each trial uses the canonical multi-turn prompt from prompt.py. For injection
trials, a concept vector is added to the residual stream at all positions
from the "Trial 1" boundary onward.

Controls are handled naturally via alpha=0 in the sweep.
Results are appended to JSONL one trial at a time for crash-safety.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from prompt import build_trial_prompt

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
VECTOR_DIR = Path("data/vectors/llama-3.3-70b-instruct")
RESULTS_DIR = Path("data/results/injected-thoughts")

ALPHAS = [0, 1, 2, 4, 8]
DEFAULT_LAYER = 40
N_TRIALS_PER_ALPHA = 50
MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0
RANDOM_SEED = 42

MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds

CONCEPTS = [
    "algorithms",
    "amphitheaters",
    "aquariums",
    "avalanches",
    "bags",
    "blood",
    "boulders",
    "bread",
    "cameras",
    "caverns",
    "constellations",
    "contraptions",
    "denim",
    "deserts",
    "dirigibles",
    "dust",
    "dynasties",
    "fountains",
    "frosts",
    "harmonies",
    "illusions",
    "information",
    "kaleidoscopes",
    "lightning",
    "masquerades",
    "memories",
    "milk",
    "mirrors",
    "monoliths",
    "oceans",
    "origami",
    "peace",
    "phones",
    "plastic",
    "poetry",
    "quarries",
    "rubber",
    "sadness",
    "satellites",
    "secrecy",
    "silver",
    "snow",
    "sugar",
    "treasures",
    "trees",
    "trumpets",
    "vegetables",
    "volcanoes",
    "xylophones",
    "youths",
]


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------
def load_concept_vector(
    vector_dir: Path, concept_slug: str, layer_idx: int
) -> torch.Tensor:
    """Load a single concept vector for a given layer."""
    path = vector_dir / f"{concept_slug}_all_layers.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(data, dict):
        vec = data[layer_idx]
    elif isinstance(data, torch.Tensor) and data.dim() == 2:
        vec = data[layer_idx]
    else:
        raise ValueError(f"Unexpected data in {path}: {type(data)}")

    return vec.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Single trial (NDIF generation with optional injection)
# ---------------------------------------------------------------------------
def run_single_trial(
    model,
    tokenizer,
    prompt: str,
    seq_len: int,
    injection_start_idx: int,
    concept_vector: torch.Tensor,
    alpha: float,
    layer: int,
) -> str:
    """Run one trial and return the decoded response.

    Adds alpha * concept_vector to the residual stream at layer `layer`,
    at all positions from injection_start_idx onward. When alpha=0 this
    is a no-op (control trial).
    """
    with model.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        remote=True,
    ) as tracer:
        if alpha > 0:
            hs = model.model.layers[layer].output[0]
            scaled = alpha * concept_vector.to(device=hs.device, dtype=hs.dtype)

            # Build an injection tensor that is zero before the boundary and
            # scaled after it. Uses explicit reassignment (not in-place +=)
            # because nnsight proxies require it.
            injection = torch.zeros_like(hs)
            injection[injection_start_idx:, :] = scaled
            model.model.layers[layer].output[0] = hs + injection

        output = tracer.result.save()

    generated_ids = output[0][seq_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_trial_with_retry(model, tokenizer, **kwargs) -> str | None:
    """Wrap run_single_trial with retries for NDIF transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_single_trial(model, tokenizer, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  Trial failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                log.error("  All retries exhausted. Skipping trial.")
                return None


# ---------------------------------------------------------------------------
# Results I/O (append-only JSONL)
# ---------------------------------------------------------------------------


def trial_key(alpha: float, concept: str, trial_idx: int) -> str:
    return f"{alpha}|{concept}|{trial_idx}"


def load_done_keys(results_path: Path) -> set[str]:
    """Read existing JSONL and return completed trial keys for resumption."""
    done: set[str] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                done.add(
                    trial_key(float(t["alpha"]), t["concept"], int(t["trial_idx"]))
                )
    return done


def append_trial(results_path: Path, record: dict) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / args.results_file
    done_keys = load_done_keys(results_path)
    if done_keys:
        log.info(f"Resuming: {len(done_keys)} trials already completed")

    # --- nnsight model setup ---
    log.info(f"Loading model: {MODEL_ID}")
    import nnsight
    from nnsight import CONFIG, LanguageModel

    if os.environ.get("NDIF_API_KEY"):
        CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

    assert nnsight.is_model_running(MODEL_ID), f"{MODEL_ID} is not online on NDIF."
    log.info("NDIF model confirmed online.")

    model = LanguageModel(MODEL_ID)
    tokenizer = model.tokenizer

    # --- Build prompt once ---
    prompt, input_ids, injection_start_idx = build_trial_prompt(tokenizer)
    seq_len = input_ids.shape[1]
    log.info(
        f"Prompt: {seq_len} tokens, injection starts at token {injection_start_idx}"
    )

    layer = args.layer
    alphas = args.alphas
    rng = random.Random(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # --- Vector cache ---
    vector_cache: dict[str, torch.Tensor] = {}

    def get_vector(concept: str) -> torch.Tensor | None:
        if concept in vector_cache:
            return vector_cache[concept]
        try:
            vec = load_concept_vector(VECTOR_DIR, concept, layer)
            vector_cache[concept] = vec
            return vec
        except Exception as e:  # noqa: BLE001
            log.warning(f"  Could not load vector for {concept}: {e}")
            return None

    # --- Trial plan ---
    # Per alpha: 50 trials (one per concept). At alpha=0 these are controls.
    total = len(alphas) * N_TRIALS_PER_ALPHA
    log.info(
        f"Total trials: {total} ({len(alphas)} alphas x {N_TRIALS_PER_ALPHA} per alpha)"
    )

    new_trials = 0
    for alpha in alphas:
        log.info(f"\n{'=' * 60}")
        log.info(f"Alpha = {alpha}, Layer = {layer}")
        log.info(f"{'=' * 60}")

        # Shuffle concepts for this alpha level
        concepts_shuffled = list(CONCEPTS)
        rng.shuffle(concepts_shuffled)

        for trial_idx, concept in enumerate(concepts_shuffled[:N_TRIALS_PER_ALPHA]):
            key = trial_key(alpha, concept, trial_idx)
            if key in done_keys:
                continue

            vec = get_vector(concept)
            if vec is None:
                continue

            label = "control" if alpha == 0 else "inject"
            log.info(f"  [{label}] alpha={alpha} concept={concept} trial={trial_idx}")

            response = run_trial_with_retry(
                model,
                tokenizer,
                prompt=prompt,
                seq_len=seq_len,
                injection_start_idx=injection_start_idx,
                concept_vector=vec,
                alpha=alpha,
                layer=layer,
            )
            if response is None:
                continue

            record = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": MODEL_ID,
                "layer": layer,
                "alpha": alpha,
                "concept": concept,
                "trial_idx": trial_idx,
                "prompt_text": prompt,
                "raw_response": response,
            }
            append_trial(results_path, record)
            done_keys.add(key)
            new_trials += 1
            log.info(f"    -> {response[:120]!r}")

    log.info(f"\nDone. {new_trials} new trials written to {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Injected-thought detection experiment"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=f"Injection layer (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=ALPHAS,
        help=f"Injection strengths (default: {ALPHAS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.jsonl",
        help="Results filename under data/results/injected-thoughts/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running",
    )
    args = parser.parse_args()

    if args.dry_run:
        total = len(args.alphas) * N_TRIALS_PER_ALPHA
        print(f"Model:           {MODEL_ID}")
        print(f"Vector dir:      {VECTOR_DIR}")
        print(f"Layer:           {args.layer}")
        print(f"Alphas:          {args.alphas}")
        print(f"Trials/alpha:    {N_TRIALS_PER_ALPHA}")
        print(f"Max tokens:      {MAX_NEW_TOKENS}")
        print(f"Temperature:     {TEMPERATURE}")
        print(f"Seed:            {args.seed}")
        print(f"Total trials:    {total}")
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
