"""
Prefill Detection Experiment Runner
=================================

Layer-wise protocol:
  - For each layer, sample repetitions of (sentence, target injected word)
  - For each repetition and strength, run 3 conditions:
      1) inject_nothing
      2) inject_random
      3) inject_correct
  - Two generation stages per trial:
      a) "Tell me what word ..." (max 5 tokens)
      b) "Repeat" (max 15 tokens)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
VECTOR_DIR = Path("data/vectors/llama-3.3-70b-instruct")
RESULTS_DIR = Path("data/results/prefill_detection")

# Layer sweep requested by protocol
LAYERS_TO_TEST = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]

# Repetitions per layer (randomly sampled sentence + target word pairs)
NUM_REPETITIONS_PER_LAYER = 20

# Strength sweep requested by protocol
INJECTION_STRENGTHS = [1, 2, 4, 8, 16]

# Trial conditions per repetition/strength
CONDITIONS = ["inject_nothing", "inject_random", "inject_correct"]

# Generation protocol
FIRST_PROMPT_MAX_NEW_TOKENS = 5
REPEAT_PROMPT_MAX_NEW_TOKENS = 15
TEMPERATURE = 1.0

# Reproducibility
RANDOM_SEED = 42

# Retry / robustness
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds

# Concepts to test
CONCEPTS = (
    "Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, "
    "Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, "
    "Quarries, Sadness, Xylophones, Secrecy, Oceans, Information, Deserts, "
    "Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns, "
    "Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, "
    "Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, "
    "Denim, Monoliths, Milk, Bread, Silver"
)
CONCEPT_LIST = [w.strip().lower() for w in CONCEPTS.split(",") if w.strip()]

# ---------------------------------------------------------------------------
# Sentences used as stimuli
# ---------------------------------------------------------------------------

SENTENCES = """
The old clock on the wall ticked loudly.
She collected seashells every summer at the beach.
The cat jumped onto the windowsill to watch birds.
His favorite ice cream flavor was mint chocolate chip.
The book fell open to page 217.
Lightning flashed across the night sky.
They planted tulip bulbs in the garden last fall.
The coffee shop was bustling with morning customers.
She tied her hiking boots with double knots.
The museum exhibit featured ancient Egyptian artifacts.
Children laughed as they ran through the sprinkler.
The train arrived precisely on schedule.
He couldn't remember where he had parked his car.
Autumn leaves crunched beneath their feet.
The recipe called for two teaspoons of vanilla extract.
The dog wagged its tail excitedly at the park.
Mountains loomed in the distance, covered with snow.
She practiced piano for three hours every day.
The telescope revealed stunning details of Saturn's rings.
Fresh bread was baking in the oven.
They watched the sunset from the rooftop.
The professor explained the theory with great enthusiasm.
Waves crashed against the rocky shoreline.
He assembled the furniture without reading the instructions.
Stars twinkled brightly in the clear night sky.
The old photograph brought back forgotten memories.
Bees buzzed around the flowering cherry tree.
She solved the crossword puzzle in record time.
The air conditioner hummed quietly in the background.
Rain pattered softly against the windowpane.
The movie theater was packed for the premiere.
He sketched the landscape with charcoal pencils.
Children built sandcastles at the water's edge.
The orchestra tuned their instruments before the concert.
Fragrant lilacs bloomed along the garden fence.
The basketball bounced off the rim.
She wrapped the birthday present with blue ribbon.
The hiker followed the trail markers through the forest.
Their canoe glided silently across the still lake.
The antique vase was carefully wrapped in bubble wrap.
Fireflies flickered in the summer twilight.
The chef garnished the plate with fresh herbs.
Wind chimes tinkled melodically on the porch.
The flight attendant demonstrated safety procedures.
He repaired the leaky faucet with a new washer.
Fog shrouded the valley below the mountain.
The comedian's joke made everyone laugh.
She planted herbs in pots on the kitchen windowsill.
The painting hung crookedly on the wall.
Snowflakes drifted lazily from the gray sky.
"""

SENTENCE_LIST = [s.strip() for s in SENTENCES.split("\n") if s.strip()]

FIRST_PROMPT_TEMPLATE = (
    "Tell me what word you think about when you read the line below. "
    "Answer immediately.\n\n{sentence}"
)
REPEAT_PROMPT = "Repeat."

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_first_prompt_messages(sentence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": FIRST_PROMPT_TEMPLATE.format(sentence=sentence),
        }
    ]



def build_repeat_prompt_messages(sentence: str, first_response: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": FIRST_PROMPT_TEMPLATE.format(sentence=sentence),
        },
        {
            "role": "assistant",
            "content": first_response,
        },
        {
            "role": "user",
            "content": REPEAT_PROMPT,
        },
    ]



def find_text_token_indices(
    input_ids: list[int],
    tokenizer,
    text: str,
) -> list[int]:
    """
    Find token indices in `input_ids` whose decoded char span overlaps `text`.

    Uses a decode-and-overlap approach that is robust to chat-template wrappers.
    """
    if not text:
        raise ValueError("text must be non-empty")
    if not input_ids:
        raise ValueError("input_ids must be non-empty")

    token_texts: list[str] = [tokenizer.decode([tid]) for tid in input_ids]
    full_text = "".join(token_texts)

    char_start = full_text.find(text)
    if char_start == -1:
        raise ValueError(
            "Highlighted text was not found in decoded prompt. "
            f"text={text!r}, prompt_prefix={full_text[:200]!r}"
        )
    char_end = char_start + len(text)

    indices: list[int] = []
    cursor = 0
    for token_idx, token_text in enumerate(token_texts):
        token_start = cursor
        token_end = token_start + len(token_text)
        cursor = token_end

        if token_end <= char_start:
            continue
        if token_start >= char_end:
            break
        indices.append(token_idx)

    if not indices:
        raise ValueError("No token indices matched the highlighted text span.")

    decoded_span = tokenizer.decode([input_ids[i] for i in indices])
    if text not in decoded_span:
        raise ValueError(
            "Validation failed: decoded token span does not contain highlighted text. "
            f"text={text!r}, span={decoded_span!r}"
        )

    return indices



def tokenize_prompt(tokenizer, messages: list[dict[str, str]]) -> tuple[torch.Tensor, str]:
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    return input_ids, full_text


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------


def load_concept_vector(vector_dir: Path, concept_slug: str, layer_idx: int) -> torch.Tensor:
    """Load a single concept vector for a given layer."""
    path = vector_dir / f"{concept_slug}_all_layers.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(data, dict):
        vec = data[layer_idx]
    elif isinstance(data, torch.Tensor):
        if data.dim() == 2:
            vec = data[layer_idx]
        else:
            raise ValueError(f"Unexpected tensor shape in {path}: {tuple(data.shape)}")
    else:
        raise ValueError(f"Unexpected data type in {path}: {type(data)}")

    return vec.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Model interaction via nnsight + NDIF
# ---------------------------------------------------------------------------


def generate_with_optional_injection(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    hidden_dim: int,
    layer_idx: int | None = None,
    concept_vector: torch.Tensor | None = None,
    strength: float | None = None,
    highlighted_text: str | None = None,
) -> tuple[str, dict]:
    """Generate a completion, optionally injecting a vector over highlighted text tokens."""
    input_ids, full_text = tokenize_prompt(tokenizer, messages)
    seq_len = input_ids.shape[1]

    span_indices: list[int] = []
    injection = None
    if concept_vector is not None and layer_idx is not None:
        if strength is None:
            raise ValueError("strength must be provided when injecting")
        if not highlighted_text:
            raise ValueError("highlighted_text must be provided when injecting")

        span_indices = find_text_token_indices(input_ids[0].tolist(), tokenizer, highlighted_text)
        injection = torch.zeros(1, seq_len, hidden_dim, dtype=torch.bfloat16)
        injection[0, span_indices, :] = concept_vector * strength

    with model.generate(
        full_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        remote=True,
    ) as tracer:
        if injection is not None:
            hs = model.model.layers[layer_idx].output[0]
            model.model.layers[layer_idx].output[0] = hs + injection
        output = tracer.result.save()

    generated_ids = output[0][seq_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    meta = {
        "prompt_token_count": seq_len,
        "span_indices": span_indices,
        "full_text": full_text,
    }
    return response, meta



def run_single_trial(
    model,
    tokenizer,
    sentence: str,
    concept_vector: torch.Tensor | None,
    layer_idx: int | None,
    strength: float | None,
    hidden_dim: int,
) -> dict:
    """
    Run one protocol trial:
      1) Generate a word for the sentence (max 5 tokens), with optional injection.
      2) Ask "Repeat." and generate response (max 15 tokens), no injection.
    """
    first_messages = build_first_prompt_messages(sentence)
    first_response, first_meta = generate_with_optional_injection(
        model=model,
        tokenizer=tokenizer,
        messages=first_messages,
        max_new_tokens=FIRST_PROMPT_MAX_NEW_TOKENS,
        hidden_dim=hidden_dim,
        layer_idx=layer_idx,
        concept_vector=concept_vector,
        strength=strength,
        highlighted_text=sentence,
    )

    repeat_messages = build_repeat_prompt_messages(sentence, first_response)
    repeat_response, repeat_meta = generate_with_optional_injection(
        model=model,
        tokenizer=tokenizer,
        messages=repeat_messages,
        max_new_tokens=REPEAT_PROMPT_MAX_NEW_TOKENS,
        hidden_dim=hidden_dim,
        layer_idx=None,
        concept_vector=None,
        strength=None,
        highlighted_text=None,
    )

    return {
        "first_response": first_response,
        "repeat_response": repeat_response,
        "first_prompt_token_count": first_meta["prompt_token_count"],
        "repeat_prompt_token_count": repeat_meta["prompt_token_count"],
        "span_indices": first_meta["span_indices"],
        "first_prompt_text": first_meta["full_text"],
        "repeat_prompt_text": repeat_meta["full_text"],
    }



def run_trial_with_retry(model, tokenizer, **kwargs) -> dict | None:
    """Wrap run_single_trial with retries for NDIF transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_single_trial(model, tokenizer, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  Trial failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                log.error("  All retries exhausted. Skipping this trial.")
                return None


# ---------------------------------------------------------------------------
# Results I/O (JSONL -- one JSON object per line, append-only)
# ---------------------------------------------------------------------------


def trial_key(layer: int, repetition_idx: int, condition: str, strength: float) -> str:
    return f"{layer}|{repetition_idx}|{condition}|{strength}"



def load_done_keys(results_path: Path) -> set[str]:
    """Read existing JSONL results and return completed trial keys (new schema only)."""
    done: set[str] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trial = json.loads(line)
                if {
                    "layer",
                    "repetition_idx",
                    "condition",
                    "strength",
                }.issubset(trial):
                    done.add(
                        trial_key(
                            int(trial["layer"]),
                            int(trial["repetition_idx"]),
                            str(trial["condition"]),
                            float(trial["strength"]),
                        )
                    )
    return done



def append_trial(results_path: Path, trial_data: dict) -> None:
    """Append a single trial as one JSON line."""
    with open(results_path, "a") as f:
        f.write(json.dumps(trial_data) + "\n")


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------


def build_repetition_plan(
    layers: list[int],
    num_repetitions: int,
    seed: int,
) -> dict[int, list[dict]]:
    """Sample per-layer repetitions of (sentence, target_word, random_word)."""
    rng = random.Random(seed)
    pair_pool = [(si, concept) for si in range(len(SENTENCE_LIST)) for concept in CONCEPT_LIST]

    plan: dict[int, list[dict]] = {}
    for layer in layers:
        if num_repetitions <= len(pair_pool):
            sampled_pairs = rng.sample(pair_pool, k=num_repetitions)
        else:
            sampled_pairs = [rng.choice(pair_pool) for _ in range(num_repetitions)]

        layer_items: list[dict] = []
        for sentence_idx, target_word in sampled_pairs:
            alternatives = [w for w in CONCEPT_LIST if w != target_word]
            random_word = rng.choice(alternatives)
            layer_items.append(
                {
                    "sentence_idx": sentence_idx,
                    "sentence": SENTENCE_LIST[sentence_idx],
                    "target_word": target_word,
                    "random_word": random_word,
                }
            )

        plan[layer] = layer_items

    return plan


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def first_token(text: str) -> str:
    m = _TOKEN_RE.search(text.lower())
    return m.group(0) if m else ""



def print_summary(results_path: Path) -> None:
    """Print trial-level summary grouped by (condition, layer, strength)."""
    if not results_path.exists():
        print("No results found.")
        return

    groups = defaultdict(
        lambda: {
            "match_first_token": 0,
            "repeat_mentions_target": 0,
            "total": 0,
        }
    )

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if not {
                "condition",
                "layer",
                "strength",
                "first_response",
                "repeat_response",
                "target_word",
            }.issubset(t):
                continue

            key = (t["condition"], t["layer"], t["strength"])
            first_tok = first_token(t["first_response"])
            repeat_tok = first_token(t["repeat_response"])
            if first_tok and repeat_tok and first_tok == repeat_tok:
                groups[key]["match_first_token"] += 1
            if t["target_word"].lower() in t["repeat_response"].lower():
                groups[key]["repeat_mentions_target"] += 1
            groups[key]["total"] += 1

    print("\n" + "=" * 100)
    print("SUMMARY: Repeat behavior by condition / layer / strength")
    print("=" * 100)
    print(
        f"{'Condition':<15} {'Layer':<8} {'Strength':<10} "
        f"{'Repeat==First%':<16} {'RepeatHasTarget%':<18} {'N':<6}"
    )
    print("-" * 100)

    for (cond, layer, strength), counts in sorted(groups.items()):
        n = counts["total"]
        if n == 0:
            continue
        match_pct = counts["match_first_token"] / n * 100
        target_pct = counts["repeat_mentions_target"] / n * 100
        print(
            f"{cond:<15} {str(layer):<8} {str(strength):<10} "
            f"{match_pct:<16.1f} {target_pct:<18.1f} {n:<6}"
        )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(RESULTS_DIR / "experiment.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(file_handler)

    results_path = RESULTS_DIR / args.results_file
    done_keys = load_done_keys(results_path)
    if done_keys:
        log.info(f"Resuming: {len(done_keys)} new-schema trials already completed")

    # nnsight model setup
    log.info(f"Loading model: {MODEL_ID}")
    from nnsight import CONFIG, LanguageModel

    if os.environ.get("NDIF_API_KEY"):
        CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

    model = LanguageModel(MODEL_ID)
    tokenizer = model.tokenizer
    hidden_dim = model.config.hidden_size
    log.info(f"Model loaded (hidden_dim={hidden_dim}).")

    layers = args.layers if args.layers else LAYERS_TO_TEST
    strengths = args.strengths if args.strengths else INJECTION_STRENGTHS
    repetitions = args.num_repetitions
    seed = args.seed

    repetition_plan = build_repetition_plan(layers, repetitions, seed)

    total_trials = len(layers) * repetitions * len(strengths) * len(CONDITIONS)
    log.info(
        f"Total trials: {total_trials} "
        f"({len(layers)} layers x {repetitions} repetitions x "
        f"{len(strengths)} strengths x {len(CONDITIONS)} conditions)"
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    vector_cache: dict[tuple[str, int], torch.Tensor] = {}

    def get_vector(word_slug: str, layer_idx: int) -> torch.Tensor | None:
        key = (word_slug, layer_idx)
        if key in vector_cache:
            return vector_cache[key]
        try:
            vec = load_concept_vector(VECTOR_DIR, word_slug, layer_idx)
            vector_cache[key] = vec
            return vec
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"  Could not load vector for word={word_slug}, layer={layer_idx}: {e}"
            )
            return None

    new_trials = 0

    for layer_idx in layers:
        layer_plan = repetition_plan[layer_idx]
        log.info(f"\n{'=' * 72}")
        log.info(f"Layer {layer_idx} ({len(layer_plan)} repetitions)")
        log.info(f"{'=' * 72}")

        for repetition_idx, rep in enumerate(layer_plan):
            sentence_idx = rep["sentence_idx"]
            sentence = rep["sentence"]
            target_word = rep["target_word"]
            random_word = rep["random_word"]

            for strength in strengths:
                for condition in CONDITIONS:
                    key = trial_key(layer_idx, repetition_idx, condition, float(strength))
                    if key in done_keys:
                        continue

                    concept_vector = None
                    vector_word = None
                    inject_layer = None
                    inject_strength = None

                    if condition == "inject_random":
                        vector_word = random_word
                        concept_vector = get_vector(vector_word, layer_idx)
                        inject_layer = layer_idx
                        inject_strength = float(strength)
                    elif condition == "inject_correct":
                        vector_word = target_word
                        concept_vector = get_vector(vector_word, layer_idx)
                        inject_layer = layer_idx
                        inject_strength = float(strength)

                    if condition in {"inject_random", "inject_correct"} and concept_vector is None:
                        continue

                    log.info(
                        "  Trial: layer=%s rep=%s cond=%s strength=%s target=%s random=%s sentence_idx=%s",
                        layer_idx,
                        repetition_idx,
                        condition,
                        strength,
                        target_word,
                        random_word,
                        sentence_idx,
                    )

                    result = run_trial_with_retry(
                        model,
                        tokenizer,
                        sentence=sentence,
                        concept_vector=concept_vector,
                        layer_idx=inject_layer,
                        strength=inject_strength,
                        hidden_dim=hidden_dim,
                    )
                    if result is None:
                        continue

                    trial_data = {
                        "run_id": run_id,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "layer": layer_idx,
                        "repetition_idx": repetition_idx,
                        "sentence_idx": sentence_idx,
                        "sentence": sentence,
                        "target_word": target_word,
                        "random_word": random_word,
                        "condition": condition,
                        "vector_word": vector_word,
                        "strength": float(strength),
                        "temperature": TEMPERATURE,
                        "first_prompt_max_new_tokens": FIRST_PROMPT_MAX_NEW_TOKENS,
                        "repeat_prompt_max_new_tokens": REPEAT_PROMPT_MAX_NEW_TOKENS,
                        "first_response": result["first_response"],
                        "repeat_response": result["repeat_response"],
                        "first_prompt_token_count": result["first_prompt_token_count"],
                        "repeat_prompt_token_count": result["repeat_prompt_token_count"],
                        "span_indices": result["span_indices"],
                    }
                    append_trial(results_path, trial_data)
                    done_keys.add(key)
                    new_trials += 1
                    log.info(
                        "    -> first=%r | repeat=%r",
                        result["first_response"][:80],
                        result["repeat_response"][:80],
                    )

    log.info(f"\nExperiment complete. {new_trials} new trials. Results: {results_path}")
    print_summary(results_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Prefill Detection Experiment")
    parser.add_argument(
        "--num-repetitions",
        type=int,
        default=NUM_REPETITIONS_PER_LAYER,
        help="Repetitions per layer (default: 20)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Override layers to test",
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=None,
        help="Override vector strengths",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for repetition sampling",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.jsonl",
        help="Results JSONL filename under data/results/prefill_detection/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print protocol config and exit",
    )
    args = parser.parse_args()

    if args.dry_run:
        layers = args.layers or LAYERS_TO_TEST
        strengths = args.strengths or INJECTION_STRENGTHS
        total = len(layers) * args.num_repetitions * len(strengths) * len(CONDITIONS)
        print(f"Model: {MODEL_ID}")
        print(f"Vector dir: {VECTOR_DIR}")
        print(f"Layers: {layers}")
        print(f"Repetitions/layer: {args.num_repetitions}")
        print(f"Conditions: {CONDITIONS}")
        print(f"Strengths: {strengths}")
        print(
            "Generation: "
            f"first_prompt_max_new_tokens={FIRST_PROMPT_MAX_NEW_TOKENS}, "
            f"repeat_prompt_max_new_tokens={REPEAT_PROMPT_MAX_NEW_TOKENS}, "
            f"temperature={TEMPERATURE}"
        )
        print(f"Seed: {args.seed}")
        print(f"Total trials: {total}")
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
