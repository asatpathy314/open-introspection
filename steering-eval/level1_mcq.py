#!/usr/bin/env python3
"""
Level 1: Multiple-Choice Propensity (Tan et al., NeurIPS 2024)

For each concept, construct an A/B question asking which topic the model is
"thinking about." Inject the concept's steering vector and measure the logit
difference between the correct and distractor option.

Run as: python level1_mcq.py [--norm raw] [--layers 20 40 60] [--concepts snow dust]

Due to nnsight 0.6 source-analysis constraints, all trace calls MUST be in
this file (not imported from another module).
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

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALPHA_SWEEP, CONCEPT_WORDS, LAYERS_TO_TEST,
    MAX_RETRIES, RESULTS_DIR, RETRY_DELAY, SEED, VECTOR_DIR,
)
from ndif_utils import setup_ndif
from vectors import get_baseline_norms, load_all_concept_vectors, normalize_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Prompt template ──

MCQ_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "Which of the following words best describes what you are thinking about right now?\n\n"
    "(A) {option_a}\n"
    "(B) {option_b}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "The answer is ("
)


def build_mcq_prompt(concept: str, distractor: str, concept_is_a: bool) -> tuple[str, str, str]:
    """Returns (prompt_text, correct_token, incorrect_token)."""
    if concept_is_a:
        prompt = MCQ_TEMPLATE.format(option_a=concept.capitalize(), option_b=distractor.capitalize())
        return prompt, "A", "B"
    else:
        prompt = MCQ_TEMPLATE.format(option_a=distractor.capitalize(), option_b=concept.capitalize())
        return prompt, "B", "A"


def run_mcq_trial(model, prompt: str, layer_idx: int, steering_vec, alpha: float) -> torch.Tensor:
    """
    Single MCQ trial: inject steering_vec at layer_idx with strength alpha,
    return logits at last token position.

    MUST be called from __main__ in this file due to nnsight source analysis.
    """
    if steering_vec is not None and alpha != 0:
        with model.trace(prompt, remote=True):
            hs = model.model.layers[layer_idx].output[0]
            sv = (alpha * steering_vec).to(device=hs.device, dtype=hs.dtype)
            hs[:] += sv
            model.model.layers[layer_idx].output[0] = hs
            logits = model.lm_head.output[0, -1, :].save()
    else:
        with model.trace(prompt, remote=True):
            logits = model.lm_head.output[0, -1, :].save()
    return logits.detach().cpu().float()


def compute_propensity(logits: torch.Tensor, tokenizer, correct_token: str, incorrect_token: str) -> float:
    """m_LD = logit(correct) - logit(incorrect)."""
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    token_map = {"A": a_id, "B": b_id}
    return (logits[token_map[correct_token]] - logits[token_map[incorrect_token]]).item()


def compute_steerability(results: list[dict]) -> dict[str, dict]:
    """Compute per-concept steerability scores (slope of propensity vs alpha)."""
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_concept[r["concept"]].append(r)

    out = {}
    for concept, rows in by_concept.items():
        # Per-sample grouping
        by_sample: dict[tuple, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            sample_key = (r["distractor"], r.get("concept_is_a", True))
            by_sample[sample_key][r["alpha"]].append(r["propensity"])

        # Aggregate
        alpha_to_props: dict[float, list[float]] = defaultdict(list)
        for r in rows:
            alpha_to_props[r["alpha"]].append(r["propensity"])

        alphas_sorted = sorted(alpha_to_props.keys())
        mean_by_alpha = {a: float(np.mean(alpha_to_props[a])) for a in alphas_sorted}

        # Fit slope
        xs = np.array(alphas_sorted)
        ys = np.array([mean_by_alpha[a] for a in alphas_sorted])
        steerability = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0

        # Per-sample slopes
        per_sample_slopes = []
        for sample_key, alpha_dict in by_sample.items():
            sa = sorted(alpha_dict.keys())
            if len(sa) < 2:
                continue
            sxs = np.array(sa)
            sys_ = np.array([np.mean(alpha_dict[a]) for a in sa])
            per_sample_slopes.append(float(np.polyfit(sxs, sys_, 1)[0]))

        anti_frac = float(np.mean([s < 0 for s in per_sample_slopes])) if per_sample_slopes else 0.0

        out[concept] = {
            "steerability": steerability,
            "mean_propensity_by_alpha": mean_by_alpha,
            "per_sample_steerabilities": per_sample_slopes,
            "anti_steerable_fraction": anti_frac,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Level 1: MCQ Propensity Sweep")
    parser.add_argument("--norm", default="raw", choices=["raw", "unit", "norm_matched"])
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--concepts", type=str, nargs="+", default=None)
    parser.add_argument("--num-distractors", type=int, default=5)
    args = parser.parse_args()

    norm = args.norm
    layers = args.layers or LAYERS_TO_TEST
    concepts = [c.lower() for c in args.concepts] if args.concepts else CONCEPT_WORDS
    alphas = ALPHA_SWEEP[norm]

    model = setup_ndif()
    tokenizer = model.tokenizer
    baseline_norms = get_baseline_norms()
    all_vecs = load_all_concept_vectors(concepts)

    # Build trial list
    rng = random.Random(SEED)
    trials = []
    for concept in concepts:
        others = [c for c in concepts if c != concept]
        distractors = rng.sample(others, min(args.num_distractors, len(others)))
        for distractor in distractors:
            concept_is_a = rng.random() < 0.5
            trials.append((concept, distractor, concept_is_a))

    output_path = RESULTS_DIR / f"level1_mcq_{norm}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["concept"], r["distractor"], r["layer"], r["alpha"]))
        log.info("Resuming: %d trials already done", len(done_keys))

    total = len(trials) * len(layers) * len(alphas)
    log.info("MCQ sweep: %d trials x %d layers x %d alphas = %d total (norm=%s)",
             len(trials), len(layers), len(alphas), total, norm)

    results = []
    completed = 0

    with open(output_path, "a") as out_f:
        for concept, distractor, concept_is_a in trials:
            prompt, correct_tok, incorrect_tok = build_mcq_prompt(concept, distractor, concept_is_a)
            concept_vec_all = all_vecs[concept]

            for layer in layers:
                raw_vec = concept_vec_all[layer]
                bl_norm = baseline_norms[layer].item()
                steering_vec = normalize_vector(raw_vec, norm, bl_norm)

                for alpha in alphas:
                    key = (concept, distractor, layer, alpha)
                    if key in done_keys:
                        completed += 1
                        continue

                    # Retry loop
                    logits = None
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            logits = run_mcq_trial(model, prompt, layer, steering_vec, alpha)
                            break
                        except Exception as e:
                            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
                            if attempt == MAX_RETRIES:
                                log.error("All retries exhausted for %s", key)
                            else:
                                time.sleep(RETRY_DELAY * attempt)

                    if logits is None:
                        continue

                    m_ld = compute_propensity(logits, tokenizer, correct_tok, incorrect_tok)

                    row = {
                        "concept": concept,
                        "distractor": distractor,
                        "concept_is_a": concept_is_a,
                        "layer": layer,
                        "alpha": alpha,
                        "norm_method": norm,
                        "propensity": m_ld,
                        "concept_vec_norm": raw_vec.norm().item(),
                        "baseline_norm": bl_norm,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    results.append(row)
                    completed += 1

                    if completed % 50 == 0:
                        log.info("  [%d/%d] %s layer=%d alpha=%.1f m_LD=%.3f",
                                 completed, total, concept, layer, alpha, m_ld)

    log.info("MCQ sweep complete. %d results -> %s", len(results), output_path)

    # Compute and save steerability
    all_results = results
    if output_path.exists():
        all_results = []
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    steerability = compute_steerability(all_results)
    summary_path = RESULTS_DIR / f"level1_steerability_{norm}.json"
    with open(summary_path, "w") as f:
        json.dump(steerability, f, indent=2, default=float)
    log.info("Steerability summary -> %s", summary_path)

    # Quick summary
    steerable = [c for c, d in steerability.items() if d["steerability"] > 0]
    log.info("Steerable: %d/%d concepts", len(steerable), len(steerability))
    for c in sorted(steerability.keys()):
        s = steerability[c]["steerability"]
        af = steerability[c]["anti_steerable_fraction"]
        log.info("  %s: steerability=%.4f, anti_frac=%.2f", c, s, af)


if __name__ == "__main__":
    main()
