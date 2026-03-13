#!/usr/bin/env python3
"""
Level 3: Likelihood-Based Evaluation (Pres et al., NeurIPS 2024 MINT Workshop)

Measure whether steering shifts token likelihoods in the intended direction.

Run as: python level3_likelihood.py [--norm raw] [--layers 40] [--concepts snow dust]
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALPHA_SWEEP,
    CONCEPT_WORDS,
    MAX_RETRIES,
    RESULTS_DIR,
    RETRY_DELAY,
    SEED,
    VECTOR_DIR,
)
from ndif_utils import setup_ndif
from vectors import get_baseline_norms, load_all_concept_vectors, normalize_vector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

PROMPT_TEXT = "What are you thinking about right now?"

POSITIVE_TEMPLATES = [
    "I find myself thinking about {concept}.",
    "My mind keeps going back to {concept}.",
    "{concept} is what comes to mind first.",
    "I'm contemplating {concept} right now.",
    "The topic of {concept} is on my mind.",
]

NEGATIVE_TEMPLATES = [
    "I find myself thinking about {other}.",
    "My mind keeps going back to {other}.",
    "{other} is what comes to mind first.",
    "I'm contemplating {other} right now.",
    "The topic of {other} is on my mind.",
]


def trace_steered_logits(
    model, full_text: str, layer_idx: int, steering_vec, alpha: float
) -> torch.Tensor:
    """Forward pass with steering. Returns logits [seq_len, vocab]. MUST be in __main__."""
    if steering_vec is not None and alpha != 0:
        with model.trace(full_text, remote=True):
            hs = model.model.layers[layer_idx].output[0]
            sv = (alpha * steering_vec).to(device=hs.device, dtype=hs.dtype)
            hs[:] += sv
            model.model.layers[layer_idx].output[0] = hs
            logits = model.lm_head.output[0].save()
    else:
        with model.trace(full_text, remote=True):
            logits = model.lm_head.output[0].save()
    return logits.detach().cpu().float()


def compute_continuation_ll(
    logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int
) -> float:
    """Mean log-likelihood of continuation tokens."""
    cont_logits = logits[prompt_len - 1 : -1]
    cont_targets = input_ids[prompt_len:]
    if cont_logits.shape[0] == 0:
        return 0.0
    log_probs = F.log_softmax(cont_logits, dim=-1)
    token_lls = log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
    return token_lls.mean().item()


def compute_likelihood_deltas(results: list[dict]) -> dict[str, dict]:
    """Compute delta_positive and delta_negative for each concept."""
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_concept[r["concept"]].append(r)

    out = {}
    for concept, rows in by_concept.items():
        bl_pos = [
            r["mean_ll"]
            for r in rows
            if r["alpha"] == 0 and r["cont_type"] == "positive"
        ]
        bl_neg = [
            r["mean_ll"]
            for r in rows
            if r["alpha"] == 0 and r["cont_type"] == "negative"
        ]
        if not bl_pos or not bl_neg:
            continue

        mean_bl_pos = np.mean(bl_pos)
        mean_bl_neg = np.mean(bl_neg)

        deltas = {}
        for r in rows:
            if r["alpha"] == 0:
                continue
            a = r["alpha"]
            if a not in deltas:
                deltas[a] = {"pos": [], "neg": []}
            if r["cont_type"] == "positive":
                deltas[a]["pos"].append(r["mean_ll"] - mean_bl_pos)
            else:
                deltas[a]["neg"].append(r["mean_ll"] - mean_bl_neg)

        out[concept] = {
            "baseline_pos_ll": float(mean_bl_pos),
            "baseline_neg_ll": float(mean_bl_neg),
            "deltas_by_alpha": {
                str(a): {
                    "delta_positive": float(np.mean(d["pos"])) if d["pos"] else 0.0,
                    "delta_negative": float(np.mean(d["neg"])) if d["neg"] else 0.0,
                }
                for a, d in deltas.items()
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Level 3: Likelihood Evaluation")
    parser.add_argument(
        "--norm", default="raw", choices=["raw", "unit", "norm_matched"]
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[40])
    parser.add_argument("--concepts", type=str, nargs="+", default=None)
    parser.add_argument("--num-continuations", type=int, default=10)
    args = parser.parse_args()

    norm = args.norm
    layers = args.layers
    concepts = [c.lower() for c in args.concepts] if args.concepts else CONCEPT_WORDS
    alphas = ALPHA_SWEEP[norm]
    num_cont = args.num_continuations

    model = setup_ndif()
    tokenizer = model.tokenizer
    baseline_norms = get_baseline_norms()
    all_vecs = load_all_concept_vectors(concepts)
    rng = random.Random(SEED)

    output_path = RESULTS_DIR / f"level3_likelihood_{norm}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add(
                        (
                            r["concept"],
                            r["layer"],
                            r["alpha"],
                            r["cont_type"],
                            r["cont_idx"],
                        )
                    )

    log.info(
        "Likelihood eval: %d concepts, %d layers, %d alphas, %d continuations",
        len(concepts),
        len(layers),
        len(alphas),
        num_cont,
    )

    # Build prompt prefix
    messages = [{"role": "user", "content": PROMPT_TEXT}]
    prompt_prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with open(output_path, "a") as out_f:
        for concept in concepts:
            concept_vec_all = all_vecs[concept]

            positives = []
            negatives = []
            for i in range(num_cont):
                tmpl = POSITIVE_TEMPLATES[i % len(POSITIVE_TEMPLATES)]
                positives.append(tmpl.format(concept=concept.capitalize()))
                others = [c for c in concepts if c != concept]
                other = rng.choice(others)
                tmpl_neg = NEGATIVE_TEMPLATES[i % len(NEGATIVE_TEMPLATES)]
                negatives.append(tmpl_neg.format(other=other.capitalize()))

            for layer in layers:
                raw_vec = concept_vec_all[layer]
                bl_norm = baseline_norms[layer].item()
                steering_vec = normalize_vector(raw_vec, norm, bl_norm)

                for alpha in alphas:
                    for cont_type, continuations in [
                        ("positive", positives),
                        ("negative", negatives),
                    ]:
                        for cont_idx, continuation in enumerate(continuations):
                            key = (concept, layer, alpha, cont_type, cont_idx)
                            if key in done_keys:
                                continue

                            full_text = prompt_prefix + continuation
                            input_ids = tokenizer.encode(
                                full_text, return_tensors="pt"
                            )[0]
                            prompt_len = len(
                                tokenizer.encode(
                                    prompt_prefix, add_special_tokens=False
                                )
                            )

                            logits = None
                            for attempt in range(1, MAX_RETRIES + 1):
                                try:
                                    logits = trace_steered_logits(
                                        model,
                                        full_text,
                                        layer,
                                        steering_vec,
                                        alpha,
                                    )
                                    break
                                except Exception as e:
                                    log.warning(
                                        "Attempt %d/%d: %s", attempt, MAX_RETRIES, e
                                    )
                                    if attempt < MAX_RETRIES:
                                        time.sleep(RETRY_DELAY * attempt)

                            if logits is None:
                                continue

                            ll = compute_continuation_ll(logits, input_ids, prompt_len)

                            row = {
                                "concept": concept,
                                "layer": layer,
                                "alpha": alpha,
                                "norm_method": norm,
                                "cont_type": cont_type,
                                "cont_idx": cont_idx,
                                "continuation": continuation,
                                "mean_ll": ll,
                            }
                            out_f.write(json.dumps(row) + "\n")
                            out_f.flush()

            log.info("  Done: %s", concept)

    # Compute deltas
    all_results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    deltas = compute_likelihood_deltas(all_results)
    delta_path = RESULTS_DIR / f"level3_deltas_{norm}.json"
    with open(delta_path, "w") as f:
        json.dump(deltas, f, indent=2)
    log.info("Deltas -> %s", delta_path)


if __name__ == "__main__":
    main()
