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
    VALIDATION_CONCEPTS,
    VALIDATION_PROMPTS,
    VECTOR_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


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


def load_token_sets_for_layer(layer: int):
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
    all_concept_toks = {}
    all_concept_toks_by_k = {}
    for concept in CONCEPT_WORDS:
        all_concept_toks[concept] = data["token_sets"][concept][layer_key][str(K_PRIMARY)]
        all_concept_toks_by_k[concept] = {}
        for k in K_VALUES:
            all_concept_toks_by_k[concept][k] = data["token_sets"][concept][layer_key][str(k)]
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

    max_k = max(K_VALUES)
    token_sets = {}       # {concept: {layer: {k: indices}}}
    token_sets_cosine = {}  # supplementary: {concept: {layer: indices}}

    for concept in CONCEPT_WORDS:
        token_sets[concept] = {}
        token_sets_cosine[concept] = {}
        for layer in layers:
            vec = all_vecs[concept][layer]

            # Primary: unembedding projection
            logit_contributions = unembed @ vec  # [vocab_size]
            topk_indices = logit_contributions.topk(max_k).indices.tolist()
            token_sets[concept][layer] = {}
            for k in K_VALUES:
                token_sets[concept][layer][k] = topk_indices[:k]

            # Supplementary: cosine similarity in unembedding space
            concept_token_ids = tokenizer.encode(concept, add_special_tokens=False)
            if concept_token_ids:
                concept_unembed = unembed[concept_token_ids[0]]
                sims = torch.cosine_similarity(unembed, concept_unembed.unsqueeze(0), dim=1)
                token_sets_cosine[concept][layer] = sims.topk(max_k).indices.tolist()

        log.info("Token sets computed for: %s", concept)

    # Baseline token set: N_BASELINE_TOKENS common tokens excluding all concept tokens
    all_concept_tokens = set()
    rep_layer = 40 if 40 in layers else layers[len(layers) // 2]
    for concept in CONCEPT_WORDS:
        if rep_layer in token_sets[concept]:
            all_concept_tokens.update(token_sets[concept][rep_layer].get(max_k, []))

    rng = random.Random(SEED)
    vocab_size = unembed.shape[0]
    # Sample from first 10K tokens (roughly the most common in LLM vocabularies)
    common_candidates = [i for i in range(min(10000, vocab_size)) if i not in all_concept_tokens]
    baseline_tokens = rng.sample(common_candidates, min(N_BASELINE_TOKENS, len(common_candidates)))

    out = {
        "token_sets": token_sets,
        "token_sets_cosine": token_sets_cosine,
        "baseline_tokens": baseline_tokens,
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
        log.info("  %s (layer %d, k=%d): %s", concept, rep_layer, K_PRIMARY, decoded)

    # Report overlap between projection and cosine methods
    overlaps = []
    for concept in CONCEPT_WORDS:
        if rep_layer in token_sets_cosine.get(concept, {}):
            proj_set = set(token_sets[concept][rep_layer][K_PRIMARY])
            cos_set = set(token_sets_cosine[concept][rep_layer][:K_PRIMARY])
            overlap = len(proj_set & cos_set) / K_PRIMARY
            overlaps.append(overlap)
    if overlaps:
        log.info("Projection vs cosine overlap (k=%d): mean=%.2f, min=%.2f, max=%.2f",
                 K_PRIMARY, np.mean(overlaps), min(overlaps), max(overlaps))


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Layer selection sweep
# ═══════════════════════════════════════════════════════════════════════════


def cmd_layer_sweep(args):
    model = setup_ndif()
    tokenizer = model.tokenizer
    all_vecs = load_all_vectors(VALIDATION_CONCEPTS)

    layers = args.layers or LAYER_SWEEP
    prompts = VALIDATION_PROMPTS
    chat_prompts = build_chat_prompts(tokenizer, prompts)

    # Load token sets
    ts_path = RESULTS_DIR / "token_sets.json"
    with open(ts_path) as f:
        ts_data = json.load(f)
    baseline_tokens = ts_data["baseline_tokens"]

    out_path = RESULTS_DIR / "layer_sweep.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = load_done_keys(
        out_path, ["concept", "prompt_idx", "layer", "alpha", "injection"]
    )

    # Count alpha=0 baselines already done (keyed by prompt_idx, layer)
    baseline_done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["alpha"] == 0:
                        baseline_done.add((r["prompt_idx"], r["layer"]))

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
                all_concept_toks[c] = ts_data["token_sets"][c][layer_key][str(K_PRIMARY)]
                all_concept_toks_by_k[c] = {}
                for k in K_VALUES:
                    all_concept_toks_by_k[c][k] = ts_data["token_sets"][c][layer_key][str(k)]

            # Phase 1: alpha=0 baselines (one pass per prompt)
            for pi, prompt_text in enumerate(chat_prompts):
                if (pi, layer) in baseline_done:
                    continue

                def _trace_baseline():
                    with model.trace(prompt_text, remote=True):
                        raw = model.lm_head.output[0, -1, :].save()
                    return raw.detach().cpu().float()

                logits = trace_with_retry(_trace_baseline)
                if logits is None:
                    log.error("Baseline failed: prompt=%d layer=%d", pi, layer)
                    continue

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
                            **metrics,
                        }
                        append_jsonl(out_f, row)

                baseline_done.add((pi, layer))

            # Phase 2: steered passes (alpha != 0)
            for concept in VALIDATION_CONCEPTS:
                vec_all = all_vecs[concept]
                vec = vec_all[layer]

                for pi, prompt_text in enumerate(chat_prompts):
                    for alpha in ALPHAS:
                        if alpha == 0:
                            continue
                        for injection in INJECTION_CONDITIONS:
                            key = (concept, pi, layer, alpha, injection)
                            if key in done_keys:
                                continue

                            sv = (alpha * vec)

                            def _trace_steered(sv=sv, injection=injection):
                                with model.trace(prompt_text, remote=True):
                                    hs = model.model.layers[layer].output[0]
                                    sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                                    if injection == "all_positions":
                                        hs[:] += sv_dev
                                    else:
                                        hs[-1, :] += sv_dev
                                    model.model.layers[layer].output[0] = hs
                                    raw = model.lm_head.output[0, -1, :].save()
                                return raw.detach().cpu().float()

                            logits = trace_with_retry(_trace_steered)
                            if logits is None:
                                log.error("Failed: %s", key)
                                continue

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
                                **metrics,
                            }
                            append_jsonl(out_f, row)
                            done_keys.add(key)

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

    model = setup_ndif()
    tokenizer = model.tokenizer
    all_vecs = load_all_vectors()
    chat_prompts = build_chat_prompts(tokenizer, NEUTRAL_PROMPTS)

    all_concept_toks, all_concept_toks_by_k, baseline_tokens = load_token_sets_for_layer(layer)

    out_path = RESULTS_DIR / f"main_sweep_layer{layer}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = load_done_keys(
        out_path, ["concept", "prompt_idx", "alpha", "injection"]
    )
    baseline_done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["alpha"] == 0:
                        baseline_done.add(r["prompt_idx"])

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

    with open(out_path, "a") as out_f:
        # Phase 1: alpha=0 baselines
        for pi, prompt_text in enumerate(chat_prompts):
            if pi in baseline_done:
                continue

            def _trace_baseline():
                with model.trace(prompt_text, remote=True):
                    raw = model.lm_head.output[0, -1, :].save()
                return raw.detach().cpu().float()

            logits = trace_with_retry(_trace_baseline)
            if logits is None:
                log.error("Baseline failed: prompt=%d", pi)
                continue

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
                        **metrics,
                    }
                    append_jsonl(out_f, row)

            baseline_done.add(pi)
            log.info("  Baseline prompt %d/%d done", pi + 1, len(chat_prompts))

        # Save baseline logits for prompt validation
        if baseline_logits:
            torch.save(baseline_logits, baseline_logits_path)
            log.info("Baseline logits saved to %s", baseline_logits_path)

        # Phase 2: steered passes
        for ci, concept in enumerate(CONCEPT_WORDS):
            vec = all_vecs[concept][layer]

            for pi, prompt_text in enumerate(chat_prompts):
                for alpha in ALPHAS:
                    if alpha == 0:
                        continue
                    for injection in INJECTION_CONDITIONS:
                        key = (concept, pi, alpha, injection)
                        if key in done_keys:
                            continue

                        sv = (alpha * vec)

                        def _trace_steered(sv=sv, injection=injection):
                            with model.trace(prompt_text, remote=True):
                                hs = model.model.layers[layer].output[0]
                                sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                                if injection == "all_positions":
                                    hs[:] += sv_dev
                                else:
                                    hs[-1, :] += sv_dev
                                model.model.layers[layer].output[0] = hs
                                raw = model.lm_head.output[0, -1, :].save()
                            return raw.detach().cpu().float()

                        logits = trace_with_retry(_trace_steered)
                        if logits is None:
                            log.error("Failed: %s", key)
                            continue

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
                            **metrics,
                        }
                        append_jsonl(out_f, row)
                        done_keys.add(key)

            log.info("  [%d/%d] %s done", ci + 1, len(CONCEPT_WORDS), concept)

    log.info("Main sweep complete -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Random control sweep
# ═══════════════════════════════════════════════════════════════════════════


def cmd_random_sweep(args):
    layer = args.layer
    if layer is None:
        raise ValueError("--layer required for random sweep")

    model = setup_ndif()
    tokenizer = model.tokenizer
    chat_prompts = build_chat_prompts(tokenizer, NEUTRAL_PROMPTS)

    all_concept_toks, all_concept_toks_by_k, baseline_tokens = load_token_sets_for_layer(layer)

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
            for pi, prompt_text in enumerate(chat_prompts):
                for alpha in ALPHAS:
                    if alpha == 0:
                        continue  # baseline shared with main sweep
                    for injection in INJECTION_CONDITIONS:
                        key = (ri, pi, alpha, injection)
                        if key in done_keys:
                            continue

                        sv = (alpha * rv)

                        def _trace_random(sv=sv, injection=injection):
                            with model.trace(prompt_text, remote=True):
                                hs = model.model.layers[layer].output[0]
                                sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                                if injection == "all_positions":
                                    hs[:] += sv_dev
                                else:
                                    hs[-1, :] += sv_dev
                                model.model.layers[layer].output[0] = hs
                                raw = model.lm_head.output[0, -1, :].save()
                            return raw.detach().cpu().float()

                        logits = trace_with_retry(_trace_random)
                        if logits is None:
                            log.error("Failed: random_%d prompt_%d alpha=%d %s",
                                      ri, pi, alpha, injection)
                            continue

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
                            "entropy": ent,
                            "cross_concept": cross,
                        }
                        append_jsonl(out_f, row)
                        done_keys.add(key)

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

    p_ms = sub.add_parser("main-sweep", help="Main experiment at optimal layer")
    p_ms.add_argument("--layer", type=int, required=True)

    p_rs = sub.add_parser("random-sweep", help="Random control sweep")
    p_rs.add_argument("--layer", type=int, required=True)

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
