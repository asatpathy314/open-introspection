#!/usr/bin/env python3
"""
Logit-Shift Steering Validation — Main Sweep Runner

IMPORTANT: nnsight 0.6 requires all trace operations in the __main__ file.
Do not refactor trace code into helper modules.

Usage:
    python run_sweep.py extract-unembed
    python run_sweep.py compute-token-sets [--layers 15 20 ...]
    python run_sweep.py layer-sweep        [--layers 15 20 ...]
    python run_sweep.py main-sweep --layer L
    python run_sweep.py random-sweep --layer L
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALPHAS,
    CONCEPT_LEXICAL_SEEDS,
    CONCEPT_WORDS,
    HIDDEN_DIM,
    INJECTION_CONDITIONS,
    K_PRIMARY,
    K_VALUES,
    LAYER_SWEEP,
    MAX_RETRIES,
    MODEL_ID,
    N_BASELINE_TOKENS,
    N_RANDOM_VECTORS,
    NEUTRAL_PROMPTS,
    NUM_LAYERS,
    RESULTS_DIR,
    RETRY_DELAY,
    SEED,
    TOKEN_SET_FAMILY_DEFAULT,
    VALIDATION_CONCEPTS,
    VALIDATION_PROMPTS,
    VECTOR_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


TOKEN_SET_FAMILY_ALIASES = {
    "projection": "projection",
    "proj": "projection",
    "lexical_cosine": "lexical_cosine",
    "lexical": "lexical_cosine",
    "independent": "lexical_cosine",
    "concept_piece_cosine": "concept_piece_cosine",
    "cosine": "concept_piece_cosine",
}


# ── Vector loading (no nnsight needed) ──


def load_concept_vector(concept: str) -> torch.Tensor:
    """Load concept vector [num_layers, hidden_dim]."""
    path = VECTOR_DIR / f"{concept.lower().replace(' ', '_')}_all_layers.pt"
    if not path.exists():
        raise FileNotFoundError(f"Vector not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True).float()


def load_all_vectors(concepts: list[str] | None = None) -> dict[str, torch.Tensor]:
    concepts = concepts or CONCEPT_WORDS
    return {c: load_concept_vector(c) for c in concepts}


# ── NDIF setup ──


def setup_ndif():
    import nnsight
    from dotenv import load_dotenv
    from nnsight import CONFIG

    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if not api_key:
        raise RuntimeError("NDIF_API_KEY not set")
    CONFIG.set_default_api_key(api_key)
    if not nnsight.is_model_running(MODEL_ID):
        raise RuntimeError(f"{MODEL_ID} is not online on NDIF")
    model = nnsight.LanguageModel(MODEL_ID)
    log.info("Model loaded: %s", MODEL_ID)
    return model


# ── Metric helpers ──


def normalize_token_text(text: str) -> str:
    """Lowercase token text with only ASCII letters retained."""
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


def is_text_like_token_text(text: str, min_alpha_chars: int = 3) -> bool:
    """Heuristic filter for plain lexical tokens in the tokenizer vocab."""
    stripped = text.lstrip()
    if not stripped:
        return False
    if "�" in stripped:
        return False
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in stripped):
        return False
    if any(ch.isdigit() for ch in stripped):
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'")
    if any(ch not in allowed for ch in stripped):
        return False
    if not (stripped == stripped.lower() or stripped == stripped.title()):
        return False
    return len(normalize_token_text(stripped)) >= min_alpha_chars


def is_word_start_token_text(text: str, min_alpha_chars: int = 3) -> bool:
    """Restrict lexical candidate pools to likely word-start tokens."""
    if not is_text_like_token_text(text, min_alpha_chars=min_alpha_chars):
        return False
    stripped = text.lstrip()
    return text.startswith(" ") or (stripped and stripped[0].isupper())


def auto_lexical_variants(concept: str) -> list[str]:
    """Simple inflectional variants for lexical seed expansion."""
    variants = [concept]
    if concept.endswith("ies") and len(concept) > 4:
        variants.append(concept[:-3] + "y")
    if concept.endswith("es") and len(concept) > 3:
        variants.append(concept[:-2])
    if concept.endswith("s") and len(concept) > 3:
        variants.append(concept[:-1])
    else:
        variants.append(concept + "s")
    out = []
    seen = set()
    for item in variants:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def lexical_seed_strings(concept: str) -> list[str]:
    """Independent lexical anchors used for the lexical cosine family."""
    seeds = []
    seen = set()
    for item in CONCEPT_LEXICAL_SEEDS.get(concept, []):
        if item not in seen:
            seeds.append(item)
            seen.add(item)
    for item in auto_lexical_variants(concept):
        if item not in seen:
            seeds.append(item)
            seen.add(item)
    return seeds


def build_vocab_text_index(tokenizer, vocab_size: int):
    """Precompute decoded token text metadata for lexical token-set construction."""
    decoded = [tokenizer.decode([tok_id]) for tok_id in range(vocab_size)]
    normalized = [normalize_token_text(text) for text in decoded]
    exact_text_index = defaultdict(list)
    text_like_token_ids = []
    for tok_id in range(vocab_size):
        norm = normalized[tok_id]
        if norm:
            exact_text_index[norm].append(tok_id)
        if is_word_start_token_text(decoded[tok_id]):
            text_like_token_ids.append(tok_id)
    return decoded, normalized, exact_text_index, text_like_token_ids


def gather_seed_token_ids(
    tokenizer,
    seed_strings: list[str],
    decoded_tokens: list[str],
    normalized_tokens: list[str],
    exact_text_index: dict[str, list[int]],
) -> list[int]:
    """Map lexical seed strings to vocab token ids, with exact-match priority."""
    seed_ids = []
    seen = set()
    for seed in seed_strings:
        norm_seed = normalize_token_text(seed)
        for tok_id in exact_text_index.get(norm_seed, []):
            if tok_id not in seen:
                seed_ids.append(tok_id)
                seen.add(tok_id)

    for seed in seed_strings:
        piece_ids = tokenizer.encode(seed, add_special_tokens=False)
        for tok_id in piece_ids:
            if tok_id in seen:
                continue
            if len(normalized_tokens[tok_id]) < 4:
                continue
            if not is_text_like_token_text(decoded_tokens[tok_id], min_alpha_chars=3):
                continue
            seed_ids.append(tok_id)
            seen.add(tok_id)

    return seed_ids


def topk_from_centroid(
    unembed: torch.Tensor,
    seed_token_ids: list[int],
    candidate_token_ids: list[int],
    normalized_tokens: list[str],
    max_k: int,
) -> list[int]:
    """Top-k text-like vocab tokens by cosine to a seed-token centroid."""
    if not seed_token_ids:
        raise ValueError("Cannot build token set without at least one seed token")
    seed_centroid = unembed[seed_token_ids].mean(dim=0)
    candidates = unembed[candidate_token_ids]
    sims = torch.cosine_similarity(candidates, seed_centroid.unsqueeze(0), dim=1)
    order = sims.argsort(descending=True).tolist()
    selected = []
    seen_norm = set()
    for idx in order:
        tok_id = candidate_token_ids[idx]
        norm = normalized_tokens[tok_id]
        if not norm or norm in seen_norm:
            continue
        selected.append(tok_id)
        seen_norm.add(norm)
        if len(selected) >= max_k:
            break
    if len(selected) < max_k:
        raise RuntimeError(
            f"Only found {len(selected)} unique normalized tokens; need {max_k}"
        )
    return selected


def canonical_token_family(family: str | None) -> str:
    raw = family or TOKEN_SET_FAMILY_DEFAULT
    try:
        return TOKEN_SET_FAMILY_ALIASES[raw]
    except KeyError as e:
        valid = ", ".join(sorted(set(TOKEN_SET_FAMILY_ALIASES.values())))
        raise ValueError(f"Unknown token family '{raw}'. Valid options: {valid}") from e


def compute_propensity(logits: torch.Tensor, concept_tokens: list[int],
                       baseline_tokens: list[int]) -> float:
    """Mean logit of concept tokens minus mean logit of baseline tokens."""
    return logits[concept_tokens].mean().item() - logits[baseline_tokens].mean().item()


def compute_entropy(logits: torch.Tensor) -> float:
    """Entropy of softmax distribution."""
    probs = torch.softmax(logits, dim=0).numpy()
    return float(scipy_entropy(probs))


def compute_cross_concept_propensities(
    logits: torch.Tensor,
    all_token_sets: dict[str, list[int]],
    baseline_tokens: list[int],
) -> dict[str, float]:
    """Propensity for every concept's token set."""
    bl_mean = logits[baseline_tokens].mean().item()
    return {c: logits[toks].mean().item() - bl_mean for c, toks in all_token_sets.items()}


def compute_supplementary_metrics(
    logits: torch.Tensor, concept_tokens: list[int]
) -> dict:
    """Rank shift and softmax probability mass for concept tokens."""
    ct = torch.tensor(concept_tokens)
    ranks = logits.argsort(descending=True).argsort()
    probs = torch.softmax(logits, dim=0)
    return {
        "mean_rank": ranks[ct].float().mean().item(),
        "prob_mass": probs[ct].sum().item(),
    }


def build_row_metrics(
    logits: torch.Tensor,
    concept: str,
    all_concept_toks: dict[str, list[int]],
    all_concept_toks_by_k: dict[str, dict[int, list[int]]],
    baseline_tokens: list[int],
) -> dict:
    """Compute all metrics for a single forward pass."""
    # Per-k own-concept propensity
    propensities = {}
    for k in K_VALUES:
        toks = all_concept_toks_by_k[concept][k]
        propensities[f"propensity_k{k}"] = compute_propensity(logits, toks, baseline_tokens)

    # Cross-concept at primary k
    cross = compute_cross_concept_propensities(logits, all_concept_toks, baseline_tokens)

    # Supplementary
    supp = compute_supplementary_metrics(logits, all_concept_toks[concept])

    return {
        **propensities,
        "entropy": compute_entropy(logits),
        "cross_concept": cross,
        **supp,
    }


# ── Token set loading ──


def _token_set_blob_for_family(data: dict, family: str) -> dict:
    family = canonical_token_family(family)
    if family == "projection":
        if "token_sets" not in data:
            raise KeyError("token_sets not found in token_sets.json")
        return data["token_sets"]
    if family == "lexical_cosine":
        key = "token_sets_lexical_cosine"
        if key not in data:
            raise KeyError(f"{key} not found in token_sets.json")
        return data[key]
    if family == "concept_piece_cosine":
        key = "token_sets_cosine"
        if key not in data:
            raise KeyError(f"{key} not found in token_sets.json")
        return data[key]
    raise ValueError(f"Unsupported token family: {family}")


def _tokens_for_k(layer_blob, k: int) -> list[int]:
    """Handle both legacy flat lists and the new {k: ids} format."""
    if not layer_blob:
        return []
    if isinstance(layer_blob, list):
        return layer_blob[:k]
    return layer_blob.get(str(k), [])


def load_token_sets_for_layer(layer: int, family: str | None = None):
    """Load precomputed token sets for a specific layer.

    Returns:
        all_concept_toks: {concept: list[int]} at K_PRIMARY
        all_concept_toks_by_k: {concept: {k: list[int]}}
        baseline_tokens: list[int]
    """
    ts_path = RESULTS_DIR / "token_sets.json"
    with open(ts_path) as f:
        data = json.load(f)
    layer_key = str(layer)
    token_blob = _token_set_blob_for_family(data, family or TOKEN_SET_FAMILY_DEFAULT)
    all_concept_toks = {}
    all_concept_toks_by_k = {}
    for concept in CONCEPT_WORDS:
        layer_blob = token_blob[concept][layer_key]
        all_concept_toks[concept] = _tokens_for_k(layer_blob, K_PRIMARY)
        all_concept_toks_by_k[concept] = {}
        for k in K_VALUES:
            all_concept_toks_by_k[concept][k] = _tokens_for_k(layer_blob, k)
    return all_concept_toks, all_concept_toks_by_k, data["baseline_tokens"]


# ── JSONL I/O ──


def load_done_keys(path: Path, key_fields: list[str]) -> set[tuple]:
    """Load existing results and return set of done keys for resume."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add(tuple(r[k] for k in key_fields))
    return done


def append_jsonl(f, row: dict):
    f.write(json.dumps(row) + "\n")
    f.flush()


# ── Retry wrapper for NDIF traces ──


def trace_with_retry(fn, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Call fn(), retrying on failure. fn should return the result."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            log.warning("Attempt %d/%d: %s", attempt, max_retries, e)
            if attempt == max_retries:
                return None
            time.sleep(delay * attempt)
    return None


# ── Chat prompt builder ──


def build_chat_prompts(tokenizer, prompts: list[str]) -> list[str]:
    """Apply chat template to each prompt."""
    result = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        result.append(text)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Extract unembedding matrix
# ═══════════════════════════════════════════════════════════════════════════


def cmd_extract_unembed(args):
    model = setup_ndif()
    log.info("Extracting unembedding matrix...")
    with model.trace("Hello", remote=True):
        unembed = model.lm_head.weight.detach().cpu().save()
    unembed = unembed.float()
    out_path = RESULTS_DIR / "unembed.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unembed, out_path)
    log.info("Saved unembedding matrix %s to %s", unembed.shape, out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Compute concept token sets
# ═══════════════════════════════════════════════════════════════════════════


def cmd_compute_token_sets(args):
    unembed_path = RESULTS_DIR / "unembed.pt"
    if not unembed_path.exists():
        raise FileNotFoundError("Run 'extract-unembed' first")
    unembed = torch.load(unembed_path, map_location="cpu", weights_only=True).float()

    model = setup_ndif()
    tokenizer = model.tokenizer
    all_vecs = load_all_vectors()
    layers = args.layers or LAYER_SWEEP

    vocab_size = unembed.shape[0]
    max_k = max(K_VALUES)
    decoded_tokens, normalized_tokens, exact_text_index, text_like_token_ids = (
        build_vocab_text_index(tokenizer, vocab_size)
    )
    token_sets = {}  # projection: {concept: {layer: {k: indices}}}
    token_sets_cosine = {}  # concept-piece cosine: {concept: {layer: {k: indices}}}
    token_sets_lexical_cosine = {}  # lexical-seed cosine: {concept: {layer: {k: indices}}}
    lexical_seed_token_ids = {}
    concept_piece_seed_token_ids = {}

    for concept in CONCEPT_WORDS:
        token_sets[concept] = {}
        token_sets_cosine[concept] = {}
        token_sets_lexical_cosine[concept] = {}

        # Independent families are computed once per concept, then repeated
        # across layers for compatibility with the existing sweep format.
        concept_piece_ids = [
            tok_id
            for tok_id in tokenizer.encode(concept, add_special_tokens=False)
            if is_text_like_token_text(decoded_tokens[tok_id], min_alpha_chars=2)
        ]
        if not concept_piece_ids:
            raise RuntimeError(f"No usable tokenizer pieces found for concept '{concept}'")
        concept_piece_seed_token_ids[concept] = concept_piece_ids
        concept_piece_topk = topk_from_centroid(
            unembed, concept_piece_ids, text_like_token_ids, normalized_tokens, max_k
        )

        lexical_seeds = lexical_seed_strings(concept)
        seed_ids = gather_seed_token_ids(
            tokenizer,
            lexical_seeds,
            decoded_tokens,
            normalized_tokens,
            exact_text_index,
        )
        if not seed_ids:
            raise RuntimeError(f"No lexical seed tokens found for concept '{concept}'")
        lexical_seed_token_ids[concept] = seed_ids
        lexical_topk = topk_from_centroid(
            unembed, seed_ids, text_like_token_ids, normalized_tokens, max_k
        )

        for layer in layers:
            vec = all_vecs[concept][layer]

            # Primary: unembedding projection
            logit_contributions = unembed @ vec  # [vocab_size]
            topk_indices = logit_contributions.topk(max_k).indices.tolist()
            token_sets[concept][layer] = {}
            token_sets_cosine[concept][layer] = {}
            token_sets_lexical_cosine[concept][layer] = {}
            for k in K_VALUES:
                token_sets[concept][layer][k] = topk_indices[:k]
                token_sets_cosine[concept][layer][k] = concept_piece_topk[:k]
                token_sets_lexical_cosine[concept][layer][k] = lexical_topk[:k]

        log.info("Token sets computed for: %s", concept)

    # Baseline token set: representative English-like tokens with typical
    # unsteered logits, excluding concept-token neighborhoods across all layers.
    all_concept_tokens = set()
    for concept in CONCEPT_WORDS:
        for layer in layers:
            if layer in token_sets[concept]:
                all_concept_tokens.update(token_sets[concept][layer].get(max_k, []))

    rng = random.Random(SEED)
    rep_layer = 40 if 40 in layers else layers[len(layers) // 2]

    # Calibrate with a few neutral prompts, then sample from text-like tokens in
    # the middle of the unsteered logit distribution instead of low-index vocab
    # ids. Low tokenizer ids are not a reliable proxy for token frequency.
    calibration_prompts = build_chat_prompts(tokenizer, VALIDATION_PROMPTS)
    calibration_logits = []

    def _trace_calibration():
        with model.trace(calibration_prompts, remote=True):
            raw = model.lm_head.output[:, -1, :].save()
        return raw.detach().cpu().float()

    logits_batch = trace_with_retry(_trace_calibration)
    if logits_batch is not None:
        calibration_logits.extend([row for row in logits_batch])

    if not calibration_logits:
        raise RuntimeError("Failed to collect baseline logits for baseline-token calibration")

    mean_calibration_logits = torch.stack(calibration_logits).mean(dim=0)

    def _is_text_like_token(token_id: int) -> bool:
        text = tokenizer.decode([token_id])
        stripped = text.lstrip()
        if not stripped:
            return False
        if "�" in stripped:
            return False
        if any(ord(ch) < 32 for ch in stripped):
            return False
        return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in stripped)

    candidate_tokens = [
        token_id
        for token_id in range(vocab_size)
        if token_id not in all_concept_tokens and _is_text_like_token(token_id)
    ]

    if len(candidate_tokens) < N_BASELINE_TOKENS:
        raise RuntimeError(
            f"Only found {len(candidate_tokens)} candidate baseline tokens; "
            f"need at least {N_BASELINE_TOKENS}"
        )

    candidate_scores = mean_calibration_logits[candidate_tokens]
    candidate_pool = []
    quantile_band = None
    for lower_q, upper_q in ((0.40, 0.60), (0.30, 0.70), (0.20, 0.80)):
        lo = torch.quantile(candidate_scores, lower_q).item()
        hi = torch.quantile(candidate_scores, upper_q).item()
        pool = [
            token_id for token_id in candidate_tokens
            if lo <= mean_calibration_logits[token_id].item() <= hi
        ]
        if len(pool) >= N_BASELINE_TOKENS:
            candidate_pool = pool
            quantile_band = [lower_q, upper_q]
            break

    if not candidate_pool:
        candidate_pool = candidate_tokens
        quantile_band = [0.0, 1.0]

    baseline_tokens = rng.sample(candidate_pool, N_BASELINE_TOKENS)

    out = {
        "token_sets": token_sets,
        "token_sets_cosine": token_sets_cosine,
        "token_sets_lexical_cosine": token_sets_lexical_cosine,
        "token_set_families": ["projection", "concept_piece_cosine", "lexical_cosine"],
        "concept_piece_seed_token_ids": concept_piece_seed_token_ids,
        "lexical_seed_strings": {
            concept: lexical_seed_strings(concept) for concept in CONCEPT_WORDS
        },
        "lexical_seed_token_ids": lexical_seed_token_ids,
        "baseline_tokens": baseline_tokens,
        "baseline_token_method": {
            "type": "neutral_logit_quantile_text_tokens",
            "n_calibration_prompts": len(calibration_logits),
            "quantile_band": quantile_band,
        },
        "k_values": K_VALUES,
        "layers": layers,
    }
    out_path = RESULTS_DIR / "token_sets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    log.info("Token sets saved to %s", out_path)

    # Print sample for inspection
    for concept in CONCEPT_WORDS[:5]:
        toks = token_sets[concept][rep_layer][K_PRIMARY]
        decoded = [tokenizer.decode([t]).strip() for t in toks]
        decoded_lex = [
            tokenizer.decode([t]).strip()
            for t in token_sets_lexical_cosine[concept][rep_layer][K_PRIMARY]
        ]
        log.info("  %s projection (layer %d, k=%d): %s", concept, rep_layer, K_PRIMARY, decoded)
        log.info("  %s lexical   (layer %d, k=%d): %s", concept, rep_layer, K_PRIMARY, decoded_lex)

    # Report overlap between projection and independent lexical methods
    overlaps = []
    for concept in CONCEPT_WORDS:
        if rep_layer in token_sets_lexical_cosine.get(concept, {}):
            proj_set = set(token_sets[concept][rep_layer][K_PRIMARY])
            lex_set = set(token_sets_lexical_cosine[concept][rep_layer][K_PRIMARY])
            overlap = len(proj_set & lex_set) / K_PRIMARY
            overlaps.append(overlap)
    if overlaps:
        log.info("Projection vs lexical overlap (k=%d): mean=%.2f, min=%.2f, max=%.2f",
                 K_PRIMARY, np.mean(overlaps), min(overlaps), max(overlaps))


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Layer selection sweep
# ═══════════════════════════════════════════════════════════════════════════


def cmd_layer_sweep(args):
    model = setup_ndif()
    tokenizer = model.tokenizer
    all_vecs = load_all_vectors(VALIDATION_CONCEPTS)
    token_family = canonical_token_family(getattr(args, "token_family", None))

    layers = args.layers or LAYER_SWEEP
    prompts = VALIDATION_PROMPTS
    chat_prompts = build_chat_prompts(tokenizer, prompts)

    # Load token sets
    ts_path = RESULTS_DIR / "token_sets.json"
    with open(ts_path) as f:
        ts_data = json.load(f)
    token_blob = _token_set_blob_for_family(ts_data, token_family)
    baseline_tokens = ts_data["baseline_tokens"]

    out_path = RESULTS_DIR / "layer_sweep.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = load_done_keys(
        out_path, ["concept", "prompt_idx", "layer", "alpha", "injection"]
    )

    # Count alpha=0 baselines already done (keyed by prompt_idx, layer).
    # Require the full concept x injection block to be present so resume does
    # not silently skip a partially written baseline prompt.
    baseline_done = set()
    expected_baseline_rows = len(VALIDATION_CONCEPTS) * len(INJECTION_CONDITIONS)
    if out_path.exists():
        baseline_counts = defaultdict(int)
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["alpha"] == 0:
                        baseline_counts[(r["prompt_idx"], r["layer"])] += 1
        baseline_done = {
            key for key, count in baseline_counts.items()
            if count >= expected_baseline_rows
        }

    n_nonzero_alphas = len([a for a in ALPHAS if a != 0])
    total_steered = (
        len(VALIDATION_CONCEPTS) * len(prompts) * n_nonzero_alphas
        * len(layers) * len(INJECTION_CONDITIONS)
    )
    total_baseline = len(prompts) * len(layers)
    log.info(
        "Layer sweep: %d baseline + %d steered passes, %d already done",
        total_baseline, total_steered, len(done_keys),
    )

    with open(out_path, "a") as out_f:
        for layer in layers:
            # Load token sets for this layer
            layer_key = str(layer)
            all_concept_toks = {}
            all_concept_toks_by_k = {}
            for c in VALIDATION_CONCEPTS:
                layer_blob = token_blob[c][layer_key]
                all_concept_toks[c] = _tokens_for_k(layer_blob, K_PRIMARY)
                all_concept_toks_by_k[c] = {}
                for k in K_VALUES:
                    all_concept_toks_by_k[c][k] = _tokens_for_k(layer_blob, k)

            # Phase 1: alpha=0 baselines (one pass per prompt)
            pending_baseline_prompts = [
                pi for pi in range(len(chat_prompts)) if (pi, layer) not in baseline_done
            ]
            if pending_baseline_prompts:
                prompt_batch = [chat_prompts[pi] for pi in pending_baseline_prompts]

                def _trace_baseline():
                    with model.trace(prompt_batch, remote=True):
                        raw = model.lm_head.output[:, -1, :].save()
                    return raw.detach().cpu().float()

                logits_batch = trace_with_retry(_trace_baseline)
                if logits_batch is None:
                    log.error("Baseline batch failed: layer=%d prompts=%s",
                              layer, pending_baseline_prompts)
                else:
                    for offset, pi in enumerate(pending_baseline_prompts):
                        logits = logits_batch[offset]

                        # Write a row for each concept x injection (all identical logits)
                        for concept in VALIDATION_CONCEPTS:
                            metrics = build_row_metrics(
                                logits, concept, all_concept_toks,
                                all_concept_toks_by_k, baseline_tokens,
                            )
                            for injection in INJECTION_CONDITIONS:
                                row = {
                                    "concept": concept,
                                    "prompt_idx": pi,
                                    "layer": layer,
                                    "alpha": 0,
                                    "injection": injection,
                                    "token_family": token_family,
                                    **metrics,
                                }
                                append_jsonl(out_f, row)

                        baseline_done.add((pi, layer))

            # Phase 2: steered passes (alpha != 0)
            for concept in VALIDATION_CONCEPTS:
                vec_all = all_vecs[concept]
                vec = vec_all[layer]

                for alpha in ALPHAS:
                    if alpha == 0:
                        continue
                    for injection in INJECTION_CONDITIONS:
                        pending_prompt_indices = [
                            pi
                            for pi in range(len(chat_prompts))
                            if (concept, pi, layer, alpha, injection) not in done_keys
                        ]
                        if not pending_prompt_indices:
                            continue

                        sv = (alpha * vec)
                        prompt_batch = [chat_prompts[pi] for pi in pending_prompt_indices]

                        def _trace_steered(sv=sv, injection=injection):
                            with model.trace(prompt_batch, remote=True):
                                hs = model.model.layers[layer].output
                                sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                                if injection == "all_positions":
                                    hs[:] += sv_dev
                                else:
                                    hs[:, -1, :] += sv_dev
                                model.model.layers[layer].output = hs
                                raw = model.lm_head.output[:, -1, :].save()
                            return raw.detach().cpu().float()

                        logits_batch = trace_with_retry(_trace_steered)
                        if logits_batch is None:
                            log.error(
                                "Failed batch: concept=%s layer=%d alpha=%d injection=%s prompts=%s",
                                concept, layer, alpha, injection, pending_prompt_indices,
                            )
                            continue

                        for offset, pi in enumerate(pending_prompt_indices):
                            logits = logits_batch[offset]
                            metrics = build_row_metrics(
                                logits, concept, all_concept_toks,
                                all_concept_toks_by_k, baseline_tokens,
                            )
                            row = {
                                "concept": concept,
                                "prompt_idx": pi,
                                "layer": layer,
                                "alpha": alpha,
                                "injection": injection,
                                "token_family": token_family,
                                **metrics,
                            }
                            append_jsonl(out_f, row)
                            done_keys.add((concept, pi, layer, alpha, injection))

                log.info("  Layer %d, concept %s done", layer, concept)

            log.info("Layer %d complete", layer)

    log.info("Layer sweep complete -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Main sweep (at optimal layer)
# ═══════════════════════════════════════════════════════════════════════════


def cmd_main_sweep(args):
    layer = args.layer
    if layer is None:
        raise ValueError("--layer required for main sweep")
    token_family = canonical_token_family(getattr(args, "token_family", None))

    model = setup_ndif()
    tokenizer = model.tokenizer
    all_vecs = load_all_vectors()
    chat_prompts = build_chat_prompts(tokenizer, NEUTRAL_PROMPTS)

    all_concept_toks, all_concept_toks_by_k, baseline_tokens = load_token_sets_for_layer(
        layer, token_family
    )

    out_path = RESULTS_DIR / f"main_sweep_layer{layer}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = load_done_keys(
        out_path, ["concept", "prompt_idx", "alpha", "injection"]
    )
    baseline_done = set()
    expected_baseline_rows = len(CONCEPT_WORDS) * len(INJECTION_CONDITIONS)
    if out_path.exists():
        baseline_counts = defaultdict(int)
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["alpha"] == 0:
                        baseline_counts[r["prompt_idx"]] += 1
        baseline_done = {
            prompt_idx for prompt_idx, count in baseline_counts.items()
            if count >= expected_baseline_rows
        }

    n_nonzero = len([a for a in ALPHAS if a != 0])
    total_steered = len(CONCEPT_WORDS) * len(NEUTRAL_PROMPTS) * n_nonzero * len(INJECTION_CONDITIONS)
    total_baseline = len(NEUTRAL_PROMPTS)
    log.info(
        "Main sweep: layer %d, %d baseline + %d steered passes, %d already done",
        layer, total_baseline, total_steered, len(done_keys),
    )

    # Save alpha=0 logits for prompt validation (cosine similarity check)
    baseline_logits_path = RESULTS_DIR / f"baseline_logits_layer{layer}.pt"
    baseline_logits = {}
    if baseline_logits_path.exists():
        try:
            baseline_logits = torch.load(
                baseline_logits_path, map_location="cpu", weights_only=True
            )
            log.info(
                "Loaded existing baseline logits for %d prompts from %s",
                len(baseline_logits), baseline_logits_path,
            )
        except Exception as e:
            log.warning("Could not load existing baseline logits from %s: %s",
                        baseline_logits_path, e)
            baseline_logits = {}

    with open(out_path, "a") as out_f:
        # Phase 1: alpha=0 baselines
        pending_baseline_prompts = [
            pi for pi in range(len(chat_prompts)) if pi not in baseline_done
        ]
        if pending_baseline_prompts:
            prompt_batch = [chat_prompts[pi] for pi in pending_baseline_prompts]

            def _trace_baseline():
                with model.trace(prompt_batch, remote=True):
                    raw = model.lm_head.output[:, -1, :].save()
                return raw.detach().cpu().float()

            logits_batch = trace_with_retry(_trace_baseline)
            if logits_batch is None:
                log.error("Baseline batch failed: prompts=%s", pending_baseline_prompts)
            else:
                for offset, pi in enumerate(pending_baseline_prompts):
                    logits = logits_batch[offset]
                    baseline_logits[pi] = logits

                    for concept in CONCEPT_WORDS:
                        metrics = build_row_metrics(
                            logits, concept, all_concept_toks,
                            all_concept_toks_by_k, baseline_tokens,
                        )
                        for injection in INJECTION_CONDITIONS:
                            row = {
                                "concept": concept,
                                "prompt_idx": pi,
                                "layer": layer,
                                "alpha": 0,
                                "injection": injection,
                                "token_family": token_family,
                                **metrics,
                            }
                            append_jsonl(out_f, row)

                    baseline_done.add(pi)

                torch.save(baseline_logits, baseline_logits_path)
                log.info("  Baseline prompts done: %s", pending_baseline_prompts)

        # Baseline logits are persisted prompt-by-prompt for resume safety.
        if baseline_logits_path.exists():
            log.info("Baseline logits saved to %s", baseline_logits_path)

        # Phase 2: steered passes
        for ci, concept in enumerate(CONCEPT_WORDS):
            vec = all_vecs[concept][layer]

            for alpha in ALPHAS:
                if alpha == 0:
                    continue
                for injection in INJECTION_CONDITIONS:
                    pending_prompt_indices = [
                        pi
                        for pi in range(len(chat_prompts))
                        if (concept, pi, alpha, injection) not in done_keys
                    ]
                    if not pending_prompt_indices:
                        continue

                    sv = (alpha * vec)
                    prompt_batch = [chat_prompts[pi] for pi in pending_prompt_indices]

                    def _trace_steered(sv=sv, injection=injection):
                        with model.trace(prompt_batch, remote=True):
                            hs = model.model.layers[layer].output
                            sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                            if injection == "all_positions":
                                hs[:] += sv_dev
                            else:
                                hs[:, -1, :] += sv_dev
                            model.model.layers[layer].output = hs
                            raw = model.lm_head.output[:, -1, :].save()
                        return raw.detach().cpu().float()

                    logits_batch = trace_with_retry(_trace_steered)
                    if logits_batch is None:
                        log.error(
                            "Failed batch: concept=%s alpha=%d injection=%s prompts=%s",
                            concept, alpha, injection, pending_prompt_indices,
                        )
                        continue

                    for offset, pi in enumerate(pending_prompt_indices):
                        logits = logits_batch[offset]
                        metrics = build_row_metrics(
                            logits, concept, all_concept_toks,
                            all_concept_toks_by_k, baseline_tokens,
                        )
                        row = {
                            "concept": concept,
                            "prompt_idx": pi,
                            "layer": layer,
                            "alpha": alpha,
                            "injection": injection,
                            "token_family": token_family,
                            **metrics,
                        }
                        append_jsonl(out_f, row)
                        done_keys.add((concept, pi, alpha, injection))

            log.info("  [%d/%d] %s done", ci + 1, len(CONCEPT_WORDS), concept)

    log.info("Main sweep complete -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Random control sweep
# ═══════════════════════════════════════════════════════════════════════════


def cmd_random_sweep(args):
    layer = args.layer
    if layer is None:
        raise ValueError("--layer required for random sweep")
    token_family = canonical_token_family(getattr(args, "token_family", None))

    model = setup_ndif()
    tokenizer = model.tokenizer
    chat_prompts = build_chat_prompts(tokenizer, NEUTRAL_PROMPTS)

    all_concept_toks, all_concept_toks_by_k, baseline_tokens = load_token_sets_for_layer(
        layer, token_family
    )

    # Generate norm-matched random vectors
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    all_vecs = load_all_vectors()
    concept_norms = [all_vecs[c][layer].norm().item() for c in CONCEPT_WORDS]

    random_vectors = []
    for _ in range(N_RANDOM_VECTORS):
        rv = torch.randn(HIDDEN_DIM)
        rv = rv / rv.norm() * rng.choice(concept_norms)
        random_vectors.append(rv)

    out_path = RESULTS_DIR / f"random_sweep_layer{layer}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = load_done_keys(
        out_path, ["random_idx", "prompt_idx", "alpha", "injection"]
    )

    n_nonzero = len([a for a in ALPHAS if a != 0])
    total = N_RANDOM_VECTORS * len(NEUTRAL_PROMPTS) * n_nonzero * len(INJECTION_CONDITIONS)
    log.info(
        "Random sweep: layer %d, %d forward passes (alpha!=0 only), %d done",
        layer, total, len(done_keys),
    )

    with open(out_path, "a") as out_f:
        for ri, rv in enumerate(random_vectors):
            for alpha in ALPHAS:
                if alpha == 0:
                    continue  # baseline shared with main sweep
                for injection in INJECTION_CONDITIONS:
                    pending_prompt_indices = [
                        pi
                        for pi in range(len(chat_prompts))
                        if (ri, pi, alpha, injection) not in done_keys
                    ]
                    if not pending_prompt_indices:
                        continue

                    sv = (alpha * rv)
                    prompt_batch = [chat_prompts[pi] for pi in pending_prompt_indices]

                    def _trace_random(sv=sv, injection=injection):
                        with model.trace(prompt_batch, remote=True):
                            hs = model.model.layers[layer].output
                            sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                            if injection == "all_positions":
                                hs[:] += sv_dev
                            else:
                                hs[:, -1, :] += sv_dev
                            model.model.layers[layer].output = hs
                            raw = model.lm_head.output[:, -1, :].save()
                        return raw.detach().cpu().float()

                    logits_batch = trace_with_retry(_trace_random)
                    if logits_batch is None:
                        log.error(
                            "Failed batch: random_%d alpha=%d %s prompts=%s",
                            ri, alpha, injection, pending_prompt_indices,
                        )
                        continue

                    for offset, pi in enumerate(pending_prompt_indices):
                        logits = logits_batch[offset]
                        ent = compute_entropy(logits)
                        cross = compute_cross_concept_propensities(
                            logits, all_concept_toks, baseline_tokens,
                        )

                        row = {
                            "random_idx": ri,
                            "random_norm": rv.norm().item(),
                            "prompt_idx": pi,
                            "layer": layer,
                            "alpha": alpha,
                            "injection": injection,
                            "token_family": token_family,
                            "entropy": ent,
                            "cross_concept": cross,
                        }
                        append_jsonl(out_f, row)
                        done_keys.add((ri, pi, alpha, injection))

            log.info("  [%d/%d] random_%d done", ri + 1, N_RANDOM_VECTORS, ri)

    log.info("Random sweep complete -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Logit-Shift Steering Validation")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("extract-unembed", help="Extract unembedding matrix from NDIF")

    p_ts = sub.add_parser("compute-token-sets", help="Compute concept token sets")
    p_ts.add_argument("--layers", type=int, nargs="+", default=None)

    p_ls = sub.add_parser("layer-sweep", help="Layer selection sweep")
    p_ls.add_argument("--layers", type=int, nargs="+", default=None)
    p_ls.add_argument("--token-family", type=str, default=TOKEN_SET_FAMILY_DEFAULT)

    p_ms = sub.add_parser("main-sweep", help="Main experiment at optimal layer")
    p_ms.add_argument("--layer", type=int, required=True)
    p_ms.add_argument("--token-family", type=str, default=TOKEN_SET_FAMILY_DEFAULT)

    p_rs = sub.add_parser("random-sweep", help="Random control sweep")
    p_rs.add_argument("--layer", type=int, required=True)
    p_rs.add_argument("--token-family", type=str, default=TOKEN_SET_FAMILY_DEFAULT)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    cmd_map = {
        "extract-unembed": cmd_extract_unembed,
        "compute-token-sets": cmd_compute_token_sets,
        "layer-sweep": cmd_layer_sweep,
        "main-sweep": cmd_main_sweep,
        "random-sweep": cmd_random_sweep,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
