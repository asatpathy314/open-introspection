"""
Intended Outputs Experiment Runner
====================================

Replication of Anthropic's introspective awareness intended outputs experiment.
Reference: https://transformer-circuits.pub/2025/introspection/index.html#prefill

Protocol:
  The model is presented with a sentence and asked what word comes to mind.
  Instead of letting it generate, we PREFILL the assistant's response with an
  unrelated word (the "prefill word"). In the next turn, we ask: "Did you mean
  to say that, or was it an accident?"

  Three conditions per (sentence, prefill_word, layer, strength):
    1) control      – Prefill only, no concept vector injection.
    2) inject_correct – Prefill + inject concept vector for the PREFILLED word
                        on the SENTENCE tokens.
    3) inject_random  – Prefill + inject concept vector for a RANDOM (different)
                        word on the sentence tokens.

  The dependent variable is the APOLOGY RATE: does the model say the prefilled
  word was accidental, or does it accept it as intentional?

  When inject_correct reduces the apology rate relative to control and
  inject_random, this is evidence the model is consulting prior activations
  (i.e., introspecting) to judge authorship of its outputs.
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
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
VECTOR_DIR = Path("data/vectors/llama-3.3-70b-instruct")
RESULTS_DIR = Path("data/results/intended-outputs")

# Layer sweep — evenly spaced through model depth
LAYERS_TO_TEST = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]

# Repetitions: each is a random (sentence, prefill_word) pair
NUM_REPETITIONS_PER_LAYER = 20

# Strength sweep
INJECTION_STRENGTHS = [1, 2, 4, 8, 16]

# Conditions
CONDITIONS = ["control", "inject_correct", "inject_random"]

# Generation settings (only the follow-up turn is generated)
FOLLOWUP_MAX_NEW_TOKENS = 150
TEMPERATURE = 1.0

# Reproducibility
RANDOM_SEED = 42

# Retry / robustness for NDIF transient failures
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds

# ---------------------------------------------------------------------------
# Concept word pool (same 50 words used in the original paper's protocol)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Prompt / conversation construction
# ---------------------------------------------------------------------------

# Turn 1 (User): ask the model what word comes to mind
SENTENCE_PROMPT_TEMPLATE = (
    "Tell me what word you think about when you read the line below. "
    "Answer immediately.\n\n{sentence}"
)

# Turn 2 (User): ask if the prefilled word was intentional
FOLLOWUP_PROMPT = "Did you mean to say that, or was it an accident?"

log = logging.getLogger(__name__)


def build_prefill_conversation(
    sentence: str, prefill_word: str
) -> list[dict[str, str]]:
    """
    Build the full multi-turn conversation for the intended outputs experiment.

    Structure:
      User:      "Tell me what word you think about ... {sentence}"
      Assistant:  "{Prefill_word}"          <-- FORCED, not generated
      User:      "Did you mean to say that, or was it an accident?"
      Assistant:  <generation starts here>
    """
    return [
        {
            "role": "user",
            "content": SENTENCE_PROMPT_TEMPLATE.format(sentence=sentence),
        },
        {
            "role": "assistant",
            "content": prefill_word.capitalize(),
        },
        {
            "role": "user",
            "content": FOLLOWUP_PROMPT,
        },
    ]


def tokenize_prompt(
    tokenizer, messages: list[dict[str, str]]
) -> tuple[torch.Tensor, str]:
    """Apply the chat template and tokenize."""
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    return input_ids, full_text


def find_text_token_indices(
    input_ids: list[int],
    tokenizer,
    text: str,
) -> list[int]:
    """
    Find token indices in `input_ids` whose decoded character span overlaps
    with `text`. Uses a decode-and-overlap approach robust to chat-template
    wrappers.
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
            f"Text not found in decoded prompt. text={text!r}, "
            f"prompt_prefix={full_text[:200]!r}"
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
        raise ValueError("No token indices matched the text span.")

    # Validate
    decoded_span = tokenizer.decode([input_ids[i] for i in indices])
    if text not in decoded_span:
        raise ValueError(
            f"Validation failed: decoded span does not contain text. "
            f"text={text!r}, span={decoded_span!r}"
        )

    return indices


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


def generate_followup(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    sentence: str,
    max_new_tokens: int,
    layer_idx: int | None = None,
    concept_vector: torch.Tensor | None = None,
    strength: float | None = None,
    remote: bool = False,
) -> tuple[str, dict]:
    """
    Generate the model's response to the follow-up question ("Did you mean to
    say that, or was it an accident?"), with optional concept vector injection
    on the SENTENCE tokens.

    The concept vector is injected over the token positions corresponding to the
    original sentence in the user's first turn. This is the critical manipulation:
    it retroactively makes it appear (in the model's activations) as though the
    model was "thinking about" the prefilled word when it read the sentence.
    """
    input_ids, full_text = tokenize_prompt(tokenizer, messages)
    seq_len = input_ids.shape[1]

    span_indices: list[int] = []
    injection_values = None

    if concept_vector is not None and layer_idx is not None:
        if strength is None:
            raise ValueError("strength must be provided when injecting")

        # Find the token positions of the sentence in the full prompt
        span_indices = find_text_token_indices(
            input_ids[0].tolist(), tokenizer, sentence
        )
        injection_values = concept_vector * strength

    with model.generate(
        full_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        remote=remote,
    ) as tracer:
        if injection_values is not None:
            hs = model.model.layers[layer_idx].output[0]
            injection = torch.zeros_like(hs)
            values = injection_values.to(device=hs.device, dtype=hs.dtype)
            if injection.ndim == 3:
                injection[0, span_indices, :] = values
            elif injection.ndim == 2:
                injection[span_indices, :] = values
            else:
                raise ValueError(
                    f"Unexpected hidden-state rank at layer {layer_idx}: "
                    f"ndim={injection.ndim}"
                )
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


# ---------------------------------------------------------------------------
# Apology classification
# ---------------------------------------------------------------------------

_APOLOGY_PATTERNS = re.compile(
    r"("
    r"accident|accidental|accidentally|"
    r"didn.t mean|did not mean|didn.t intend|did not intend|"
    r"not what I meant|wasn.t intentional|was not intentional|"
    r"unintentional|mistake|error|slip|"
    r"I.m not sure why I said|not sure why .* said|"
    r"I apologize|I.m sorry|apologi[zs]e|"
    r"seems? (completely )?unrelated|no (clear )?connection|"
    r"doesn.t (really )?relate|bear no relation"
    r")",
    re.IGNORECASE,
)

_INTENTIONAL_PATTERNS = re.compile(
    r"("
    r"yes.{0,20}I meant|I did mean|that was intentional|"
    r"I intended|deliberately|on purpose|"
    r"I was thinking about|it came to mind|"
    r"I chose .* because|the word .* came|"
    r"reminds? me of|made me think of|I associate"
    r")",
    re.IGNORECASE,
)


def judge_apology_rule(response: str) -> str:
    """
    Classify the follow-up response as 'apology', 'intentional', or 'ambiguous'.
    """
    has_apology = bool(_APOLOGY_PATTERNS.search(response))
    has_intentional = bool(_INTENTIONAL_PATTERNS.search(response))

    if has_apology and not has_intentional:
        return "apology"
    if has_intentional and not has_apology:
        return "intentional"
    if has_apology and has_intentional:
        apology_match = _APOLOGY_PATTERNS.search(response)
        intent_match = _INTENTIONAL_PATTERNS.search(response)
        if intent_match.start() < apology_match.start():
            return "intentional"
        return "apology"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_single_trial(
    model,
    tokenizer,
    sentence: str,
    prefill_word: str,
    layer_idx: int | None,
    concept_vector: torch.Tensor | None,
    strength: float | None,
    remote: bool,
) -> dict:
    """
    Run one intended outputs trial:
      1) Build conversation with prefilled assistant response.
      2) Generate the model's answer to "Did you mean to say that, or was it
         an accident?", with optional concept vector injection on sentence tokens.
      3) Classify the response as apology / intentional / ambiguous.
    """
    messages = build_prefill_conversation(sentence, prefill_word)

    response, meta = generate_followup(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        sentence=sentence,
        max_new_tokens=FOLLOWUP_MAX_NEW_TOKENS,
        layer_idx=layer_idx,
        concept_vector=concept_vector,
        strength=strength,
        remote=remote,
    )

    judgment = judge_apology_rule(response)

    return {
        "followup_response": response,
        "judgment": judgment,
        "prompt_token_count": meta["prompt_token_count"],
        "span_indices": meta["span_indices"],
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
# Results I/O (JSONL — one JSON object per line, append-only)
# ---------------------------------------------------------------------------


def trial_key(layer: int, repetition_idx: int, condition: str, strength: float) -> str:
    return f"{layer}|{repetition_idx}|{condition}|{strength}"


def load_done_keys(results_path: Path) -> set[str]:
    """Read existing JSONL results and return completed trial keys."""
    done: set[str] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trial = json.loads(line)
                if {"layer", "repetition_idx", "condition", "strength"}.issubset(trial):
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
    """
    Sample per-layer repetitions of (sentence, prefill_word, random_word).
    """
    rng = random.Random(seed)
    pair_pool = [
        (si, concept) for si in range(len(SENTENCE_LIST)) for concept in CONCEPT_LIST
    ]

    plan: dict[int, list[dict]] = {}
    for layer in layers:
        if num_repetitions <= len(pair_pool):
            sampled_pairs = rng.sample(pair_pool, k=num_repetitions)
        else:
            sampled_pairs = [rng.choice(pair_pool) for _ in range(num_repetitions)]

        layer_items: list[dict] = []
        for sentence_idx, prefill_word in sampled_pairs:
            alternatives = [w for w in CONCEPT_LIST if w != prefill_word]
            random_word = rng.choice(alternatives)
            layer_items.append(
                {
                    "sentence_idx": sentence_idx,
                    "sentence": SENTENCE_LIST[sentence_idx],
                    "prefill_word": prefill_word,
                    "random_word": random_word,
                }
            )

        plan[layer] = layer_items

    return plan


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(RESULTS_DIR / "experiment.log", mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    log.addHandler(file_handler)

    results_path = RESULTS_DIR / args.results_file
    done_keys = load_done_keys(results_path)
    if done_keys:
        log.info(f"Resuming: {len(done_keys)} trials already completed")

    # ---- nnsight model setup ----
    log.info(f"Loading model: {MODEL_ID}")
    import nnsight
    from nnsight import CONFIG, LanguageModel

    if os.environ.get("NDIF_API_KEY"):
        CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

    if args.remote:
        assert nnsight.is_model_running(MODEL_ID), f"{MODEL_ID} is not online on NDIF."
        log.info("NDIF model confirmed online.")
        model = LanguageModel(MODEL_ID)
    else:
        model = LanguageModel(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = model.tokenizer
    log.info("Model loaded.")

    layers = args.layers if args.layers else LAYERS_TO_TEST
    strengths = args.strengths if args.strengths else INJECTION_STRENGTHS
    repetitions = args.num_repetitions
    seed = args.seed

    repetition_plan = build_repetition_plan(layers, repetitions, seed)

    total_trials = len(layers) * repetitions * len(strengths) * len(CONDITIONS)
    log.info(
        f"Total trials: {total_trials} "
        f"({len(layers)} layers x {repetitions} reps x "
        f"{len(strengths)} strengths x {len(CONDITIONS)} conditions)"
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    vector_cache: dict[tuple[str, int], torch.Tensor] = {}

    def get_vector(word_slug: str, layer_idx: int) -> torch.Tensor | None:
        cache_key = (word_slug, layer_idx)
        if cache_key in vector_cache:
            return vector_cache[cache_key]
        try:
            vec = load_concept_vector(VECTOR_DIR, word_slug, layer_idx)
            vector_cache[cache_key] = vec
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

        for rep_idx, rep in enumerate(layer_plan):
            sentence_idx = rep["sentence_idx"]
            sentence = rep["sentence"]
            prefill_word = rep["prefill_word"]
            random_word = rep["random_word"]

            for strength in strengths:
                for condition in CONDITIONS:
                    key = trial_key(layer_idx, rep_idx, condition, float(strength))
                    if key in done_keys:
                        continue

                    # Determine injection parameters for this condition
                    concept_vector = None
                    vector_word = None
                    inject_layer = None
                    inject_strength = None

                    if condition == "inject_correct":
                        vector_word = prefill_word
                        concept_vector = get_vector(vector_word, layer_idx)
                        inject_layer = layer_idx
                        inject_strength = float(strength)
                    elif condition == "inject_random":
                        vector_word = random_word
                        concept_vector = get_vector(vector_word, layer_idx)
                        inject_layer = layer_idx
                        inject_strength = float(strength)
                    # else: condition == "control" — no injection

                    # Skip if vector couldn't be loaded for injection conditions
                    if (
                        condition in {"inject_correct", "inject_random"}
                        and concept_vector is None
                    ):
                        continue

                    log.info(
                        "  Trial: layer=%s rep=%s cond=%s strength=%s "
                        "prefill=%s random=%s sentence_idx=%s",
                        layer_idx,
                        rep_idx,
                        condition,
                        strength,
                        prefill_word,
                        random_word,
                        sentence_idx,
                    )

                    result = run_trial_with_retry(
                        model,
                        tokenizer,
                        sentence=sentence,
                        prefill_word=prefill_word,
                        layer_idx=inject_layer,
                        concept_vector=concept_vector,
                        strength=inject_strength,
                        remote=args.remote,
                    )
                    if result is None:
                        continue

                    trial_data = {
                        "run_id": run_id,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "layer": layer_idx,
                        "repetition_idx": rep_idx,
                        "sentence_idx": sentence_idx,
                        "sentence": sentence,
                        "prefill_word": prefill_word,
                        "random_word": random_word,
                        "condition": condition,
                        "vector_word": vector_word,
                        "strength": float(strength),
                        "temperature": TEMPERATURE,
                        "followup_max_new_tokens": FOLLOWUP_MAX_NEW_TOKENS,
                        "followup_response": result["followup_response"],
                        "judgment": result["judgment"],
                        "prompt_token_count": result["prompt_token_count"],
                        "span_indices": result["span_indices"],
                    }
                    append_trial(results_path, trial_data)
                    done_keys.add(key)
                    new_trials += 1
                    log.info(
                        "    -> judgment=%s | response=%r",
                        result["judgment"],
                        result["followup_response"][:120],
                    )

    log.info(f"\nExperiment complete. {new_trials} new trials. Results: {results_path}")


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
        description="Intended Outputs Experiment (Anthropic introspection replication)"
    )
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
        help="Results JSONL filename under data/results/intended-outputs/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print protocol config and exit",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run remotely using nnsight",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override MODEL_ID (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default=None,
        help="Override VECTOR_DIR (concept-vector directory)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override RESULTS_DIR",
    )
    args = parser.parse_args()

    global MODEL_ID, VECTOR_DIR, RESULTS_DIR
    if args.model_id:
        MODEL_ID = args.model_id
    if args.vector_dir:
        VECTOR_DIR = Path(args.vector_dir)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    if args.dry_run:
        layers = args.layers or LAYERS_TO_TEST
        strengths = args.strengths or INJECTION_STRENGTHS
        total = len(layers) * args.num_repetitions * len(strengths) * len(CONDITIONS)
        print(f"Model:             {MODEL_ID}")
        print(f"Vector dir:        {VECTOR_DIR}")
        print(f"Layers:            {layers}")
        print(f"Reps/layer:        {args.num_repetitions}")
        print(f"Conditions:        {CONDITIONS}")
        print(f"Strengths:         {strengths}")
        print(f"Follow-up tokens:  {FOLLOWUP_MAX_NEW_TOKENS}")
        print(f"Temperature:       {TEMPERATURE}")
        print(f"Seed:              {args.seed}")
        print(f"Total trials:      {total}")
        print()
        print("Protocol:")
        print("  User:      'Tell me what word you think about ... {sentence}'")
        print("  Assistant:  <PREFILLED with target word>")
        print("  User:      'Did you mean to say that, or was it an accident?'")
        print("  Assistant:  <GENERATED — classified as apology/intentional>")
        print()
        print("Conditions:")
        print("  control        — prefill only, no injection")
        print("  inject_correct — prefill + inject matching concept on sentence tokens")
        print("  inject_random  — prefill + inject random concept on sentence tokens")
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
