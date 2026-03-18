#!/usr/bin/env python3
"""
Logit-Shift Steering Validation — Statistical Analysis

Usage:
    python analysis.py --layer L                    # Full analysis at layer L
    python analysis.py --layer-sweep                # Analyze layer sweep only
    python analysis.py --validate-prompts --layer L # Prompt validation checks
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ABSTRACT_CONCEPTS,
    ALPHAS,
    CONCEPT_WORDS,
    CONCRETE_CONCEPTS,
    INJECTION_CONDITIONS,
    K_PRIMARY,
    K_VALUES,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── I/O ──


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def _json_convert(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ── BH FDR correction ──


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= rank / n * q:
            max_k = rank
    if max_k > 0:
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_k:
                significant[orig_idx] = True
    return significant


# ═══════════════════════════════════════════════════════════════════════════
# Layer Selection
# ═══════════════════════════════════════════════════════════════════════════


def analyze_layer_sweep(sweep_path: Path) -> dict:
    """Analyze layer sweep results and identify optimal layer per injection condition."""
    results = load_jsonl(sweep_path)

    # Group by (layer, injection, concept, prompt_idx) -> [(alpha, propensity)]
    grouped = defaultdict(list)
    for r in results:
        key = (r["layer"], r["injection"], r["concept"], r["prompt_idx"])
        grouped[key].append((r["alpha"], r[f"propensity_k{K_PRIMARY}"]))

    # Compute per-(concept, prompt) slopes, grouped by (layer, injection)
    layer_inj_slopes = defaultdict(list)
    for (layer, injection, concept, prompt_idx), points in grouped.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if len(xs) >= 2:
            slope = np.polyfit(xs, ys, 1)[0]
            layer_inj_slopes[(layer, injection)].append(float(slope))

    # Summary per (layer, injection)
    layer_scores = {}
    for (layer, injection), slopes in sorted(layer_inj_slopes.items()):
        key = f"layer{layer}_{injection}"
        layer_scores[key] = {
            "layer": layer,
            "injection": injection,
            "mean_steerability": float(np.mean(slopes)),
            "median_steerability": float(np.median(slopes)),
            "std": float(np.std(slopes)),
            "n_slopes": len(slopes),
        }

    # Best layer per injection condition
    best = {}
    for injection in INJECTION_CONDITIONS:
        candidates = {
            k: v for k, v in layer_scores.items() if v["injection"] == injection
        }
        if candidates:
            best_key = max(candidates, key=lambda k: candidates[k]["mean_steerability"])
            best[injection] = candidates[best_key]

    return {"layer_scores": layer_scores, "best": best}


# ═══════════════════════════════════════════════════════════════════════════
# Steerability Computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_steerability(
    results: list[dict], injection: str, propensity_key: str = f"propensity_k{K_PRIMARY}"
) -> dict[str, dict]:
    """Compute per-concept steerability from sweep results."""
    filtered = [r for r in results if r["injection"] == injection]

    # Group by (concept, prompt_idx) -> [(alpha, propensity)]
    by_cp = defaultdict(lambda: defaultdict(list))
    for r in filtered:
        by_cp[r["concept"]][r["prompt_idx"]].append((r["alpha"], r[propensity_key]))

    steerability = {}
    for concept in CONCEPT_WORDS:
        per_prompt_slopes = []
        for pi, points in sorted(by_cp[concept].items()):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if len(xs) >= 2:
                slope = float(np.polyfit(xs, ys, 1)[0])
                per_prompt_slopes.append(slope)

        mean_slope = float(np.mean(per_prompt_slopes)) if per_prompt_slopes else 0.0

        # Statistical test
        if len(per_prompt_slopes) >= 8:
            _, shapiro_p = stats.shapiro(per_prompt_slopes)
            if shapiro_p < 0.05:
                # Non-normal: Wilcoxon signed-rank test
                try:
                    stat_val, p_val = stats.wilcoxon(
                        per_prompt_slopes, alternative="greater"
                    )
                    test_used = "wilcoxon"
                except ValueError:
                    p_val = 1.0
                    test_used = "wilcoxon_failed"
            else:
                t_stat, p_two = stats.ttest_1samp(per_prompt_slopes, 0)
                p_val = p_two / 2 if t_stat > 0 else 1 - p_two / 2
                test_used = "t-test"
        elif len(per_prompt_slopes) >= 2:
            t_stat, p_two = stats.ttest_1samp(per_prompt_slopes, 0)
            p_val = p_two / 2 if t_stat > 0 else 1 - p_two / 2
            test_used = "t-test"
        else:
            p_val = 1.0
            test_used = "insufficient_data"

        # Vector norm (load from disk)
        try:
            from config import VECTOR_DIR
            vec_path = VECTOR_DIR / f"{concept.lower().replace(' ', '_')}_all_layers.pt"
            vec = torch.load(vec_path, map_location="cpu", weights_only=True).float()
            # Use the layer from the first result
            layer = filtered[0]["layer"] if filtered else 40
            vec_norm = float(vec[layer].norm().item())
        except Exception:
            vec_norm = None

        steerability[concept] = {
            "steerability": mean_slope,
            "per_prompt_slopes": per_prompt_slopes,
            "p_value": float(p_val),
            "test_used": test_used,
            "n_prompts": len(per_prompt_slopes),
            "vec_norm": vec_norm,
            "category": "abstract" if concept in ABSTRACT_CONCEPTS else "concrete",
        }

    return steerability


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Concept Specificity Matrix
# ═══════════════════════════════════════════════════════════════════════════


def compute_specificity_matrix(
    results: list[dict], injection: str, alpha: float | None = None
) -> dict[str, dict[str, float]]:
    """Compute 50x50 cross-concept specificity matrix.

    Returns {injected_concept: {target_concept: mean_propensity}}.
    """
    filtered = [r for r in results if r["injection"] == injection]
    if alpha is not None:
        filtered = [r for r in filtered if r["alpha"] == alpha]

    agg = defaultdict(lambda: defaultdict(list))
    for r in filtered:
        for target, prop in r["cross_concept"].items():
            agg[r["concept"]][target].append(prop)

    matrix = {}
    for injected in CONCEPT_WORDS:
        if injected in agg:
            matrix[injected] = {
                t: float(np.mean(vs)) for t, vs in agg[injected].items()
            }
    return matrix


def row_normalize_matrix(matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Divide each row by its diagonal entry."""
    normalized = {}
    for injected, row in matrix.items():
        diag = row.get(injected, 1.0)
        if abs(diag) < 1e-10:
            diag = 1e-10
        normalized[injected] = {t: v / diag for t, v in row.items()}
    return normalized


def specificity_diagonal_dominance(matrix: dict[str, dict[str, float]]) -> dict:
    """Test whether diagonal entries dominate off-diagonal (paired t-test)."""
    diag_vals = []
    off_diag_means = []
    for concept in CONCEPT_WORDS:
        if concept not in matrix:
            continue
        row = matrix[concept]
        diag = row.get(concept, 0)
        off = [v for t, v in row.items() if t != concept and t in CONCEPT_WORDS]
        if off:
            diag_vals.append(diag)
            off_diag_means.append(np.mean(off))

    if len(diag_vals) >= 2:
        t_stat, p_val = stats.ttest_rel(diag_vals, off_diag_means)
        return {
            "mean_diagonal": float(np.mean(diag_vals)),
            "mean_off_diagonal": float(np.mean(off_diag_means)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "n": len(diag_vals),
        }
    return {"error": "insufficient_data"}


# ═══════════════════════════════════════════════════════════════════════════
# Real vs Random Comparison
# ═══════════════════════════════════════════════════════════════════════════


def compare_real_vs_random(
    main_results: list[dict], random_results: list[dict], injection: str
) -> dict:
    """Compare real concept steerability vs random vector steerability."""
    real_steer = compute_steerability(main_results, injection)

    # For random: group by (random_idx, target_concept, prompt_idx) -> [(alpha, propensity)]
    random_filtered = [r for r in random_results if r["injection"] == injection]

    random_by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in random_filtered:
        for target, prop in r["cross_concept"].items():
            random_by[r["random_idx"]][target][r["prompt_idx"]].append(
                (r["alpha"], prop)
            )

    # Include alpha=0 from main sweep baseline for random steerability calculation
    main_baseline = [r for r in main_results if r["injection"] == injection and r["alpha"] == 0]
    baseline_propensity = defaultdict(lambda: defaultdict(list))
    for r in main_baseline:
        for target, prop in r["cross_concept"].items():
            baseline_propensity[target][r["prompt_idx"]].append(prop)

    # Compute steerability for each (random_idx, target_concept)
    random_steerabilities = defaultdict(list)  # {target_concept: [mean_slopes]}
    for ri, by_target in random_by.items():
        for target, by_prompt in by_target.items():
            slopes = []
            for pi, points in by_prompt.items():
                # Add alpha=0 point from main baseline
                all_points = list(points)
                if target in baseline_propensity and pi in baseline_propensity[target]:
                    bl_vals = baseline_propensity[target][pi]
                    if bl_vals:
                        all_points.append((0, float(np.mean(bl_vals))))

                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                if len(xs) >= 2:
                    slopes.append(float(np.polyfit(xs, ys, 1)[0]))
            if slopes:
                random_steerabilities[target].append(float(np.mean(slopes)))

    comparison = {}
    n_exceeding = 0
    for concept in CONCEPT_WORDS:
        real_s = real_steer[concept]["steerability"]
        rand_s = random_steerabilities.get(concept, [0.0])
        p95 = float(np.percentile(rand_s, 95)) if len(rand_s) > 1 else max(rand_s)
        exceeds = real_s > p95
        if exceeds:
            n_exceeding += 1
        comparison[concept] = {
            "real_steerability": real_s,
            "random_95th": p95,
            "random_mean": float(np.mean(rand_s)),
            "random_std": float(np.std(rand_s)),
            "exceeds_95th": exceeds,
        }

    return {
        "per_concept": comparison,
        "n_exceeding_95th": n_exceeding,
        "frac_exceeding_95th": n_exceeding / max(len(CONCEPT_WORDS), 1),
        "all_random_steerabilities": {
            c: vs for c, vs in random_steerabilities.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Entropy Flagging
# ═══════════════════════════════════════════════════════════════════════════


def flag_entropy_breakdown(results: list[dict], injection: str,
                           threshold_factor: float = 3.0) -> dict:
    """Flag data points where entropy deviates too far from alpha=0 baseline."""
    filtered = [r for r in results if r["injection"] == injection]

    # Baseline entropy per prompt
    baseline_entropy = defaultdict(list)
    for r in filtered:
        if r["alpha"] == 0:
            baseline_entropy[r["prompt_idx"]].append(r["entropy"])
    baseline_mean = {pi: np.mean(vs) for pi, vs in baseline_entropy.items()}

    flagged = []
    total_steered = 0
    for r in filtered:
        if r["alpha"] == 0:
            continue
        total_steered += 1
        bl = baseline_mean.get(r["prompt_idx"])
        if bl is None or bl == 0:
            continue
        ratio = r["entropy"] / bl
        if ratio < 1 / threshold_factor or ratio > threshold_factor:
            flagged.append({
                "concept": r["concept"],
                "prompt_idx": r["prompt_idx"],
                "alpha": r["alpha"],
                "entropy": r["entropy"],
                "baseline_entropy": bl,
                "ratio": ratio,
            })

    return {
        "n_flagged": len(flagged),
        "n_total": total_steered,
        "frac_flagged": len(flagged) / max(total_steered, 1),
        "flagged_samples": flagged[:50],  # cap for readability
    }


# ═══════════════════════════════════════════════════════════════════════════
# K Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════


def k_sensitivity_analysis(results: list[dict], injection: str) -> dict:
    """Compute steerability at each k value and report consistency."""
    out = {}
    for k in K_VALUES:
        key = f"propensity_k{k}"
        # Check that this key exists in data
        if results and key not in results[0]:
            out[k] = {"error": f"key {key} not found in results"}
            continue
        steer = compute_steerability(results, injection, propensity_key=key)
        slopes = [steer[c]["steerability"] for c in CONCEPT_WORDS]
        out[k] = {
            "mean_steerability": float(np.mean(slopes)),
            "median_steerability": float(np.median(slopes)),
            "std": float(np.std(slopes)),
            "per_concept": {c: steer[c]["steerability"] for c in CONCEPT_WORDS},
        }

    # Pairwise correlation between k values
    correlations = {}
    ks = [k for k in K_VALUES if k in out and "per_concept" in out[k]]
    for i, k1 in enumerate(ks):
        for k2 in ks[i + 1:]:
            v1 = [out[k1]["per_concept"][c] for c in CONCEPT_WORDS]
            v2 = [out[k2]["per_concept"][c] for c in CONCEPT_WORDS]
            rho, p = stats.spearmanr(v1, v2)
            correlations[f"k{k1}_vs_k{k2}"] = {
                "spearman_rho": float(rho), "p_value": float(p)
            }

    return {"per_k": out, "correlations": correlations}


# ═══════════════════════════════════════════════════════════════════════════
# Abstract vs Concrete Breakdown
# ═══════════════════════════════════════════════════════════════════════════


def abstract_vs_concrete(steerability: dict[str, dict]) -> dict:
    """Compare steerability between abstract and concrete concepts."""
    abstract_slopes = [
        steerability[c]["steerability"]
        for c in CONCEPT_WORDS if c in ABSTRACT_CONCEPTS and c in steerability
    ]
    concrete_slopes = [
        steerability[c]["steerability"]
        for c in CONCEPT_WORDS if c in CONCRETE_CONCEPTS and c in steerability
    ]

    result = {
        "abstract": {
            "n": len(abstract_slopes),
            "mean": float(np.mean(abstract_slopes)) if abstract_slopes else 0,
            "std": float(np.std(abstract_slopes)) if abstract_slopes else 0,
            "median": float(np.median(abstract_slopes)) if abstract_slopes else 0,
        },
        "concrete": {
            "n": len(concrete_slopes),
            "mean": float(np.mean(concrete_slopes)) if concrete_slopes else 0,
            "std": float(np.std(concrete_slopes)) if concrete_slopes else 0,
            "median": float(np.median(concrete_slopes)) if concrete_slopes else 0,
        },
    }

    # Mann-Whitney U test for group difference
    if len(abstract_slopes) >= 2 and len(concrete_slopes) >= 2:
        u_stat, p_val = stats.mannwhitneyu(
            concrete_slopes, abstract_slopes, alternative="greater"
        )
        result["mann_whitney"] = {"u_stat": float(u_stat), "p_value": float(p_val)}

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Validation
# ═══════════════════════════════════════════════════════════════════════════


def validate_prompts(layer: int) -> dict:
    """Run prompt validation checks on alpha=0 baseline data."""
    out = {}

    # Check 1: No concept token set has anomalously high logits at alpha=0
    main_path = RESULTS_DIR / f"main_sweep_layer{layer}.jsonl"
    if main_path.exists():
        results = load_jsonl(main_path)
        baseline = [r for r in results if r["alpha"] == 0]

        # Mean propensity per concept at alpha=0 across all prompts
        concept_baseline_prop = defaultdict(list)
        for r in baseline:
            concept_baseline_prop[r["concept"]].append(r[f"propensity_k{K_PRIMARY}"])

        baseline_stats = {}
        props = []
        for c in CONCEPT_WORDS:
            vals = concept_baseline_prop.get(c, [])
            if vals:
                mean_p = float(np.mean(vals))
                props.append(mean_p)
                baseline_stats[c] = {"mean_propensity_alpha0": mean_p}

        # Flag if any concept > 2 std above the mean
        if props:
            overall_mean = np.mean(props)
            overall_std = np.std(props)
            for c in CONCEPT_WORDS:
                if c in baseline_stats:
                    z = (baseline_stats[c]["mean_propensity_alpha0"] - overall_mean) / max(overall_std, 1e-10)
                    baseline_stats[c]["z_score"] = float(z)
                    baseline_stats[c]["anomalous"] = abs(z) > 2

        out["concept_baseline_propensities"] = baseline_stats
        n_anomalous = sum(1 for v in baseline_stats.values() if v.get("anomalous", False))
        out["n_anomalous_concepts"] = n_anomalous

    # Check 2: Pairwise cosine similarity of baseline logit distributions
    baseline_logits_path = RESULTS_DIR / f"baseline_logits_layer{layer}.pt"
    if baseline_logits_path.exists():
        bl_logits = torch.load(baseline_logits_path, map_location="cpu", weights_only=True)
        prompt_indices = sorted(bl_logits.keys())
        n = len(prompt_indices)
        if n >= 2:
            # Compute pairwise cosine similarity
            vecs = torch.stack([bl_logits[i] for i in prompt_indices])
            vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
            cos_sim = (vecs_norm @ vecs_norm.T).numpy()

            # Upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n, k=1)
            pairwise = cos_sim[triu_indices]

            out["prompt_cosine_similarity"] = {
                "mean": float(np.mean(pairwise)),
                "std": float(np.std(pairwise)),
                "min": float(np.min(pairwise)),
                "max": float(np.max(pairwise)),
                "is_degenerate": float(np.min(pairwise)) > 0.99,
            }

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Token Set Quality Report
# ═══════════════════════════════════════════════════════════════════════════


def token_set_quality_report() -> dict:
    """Report overlap between projection and cosine token sets."""
    ts_path = RESULTS_DIR / "token_sets.json"
    if not ts_path.exists():
        return {"error": "token_sets.json not found"}

    with open(ts_path) as f:
        data = json.load(f)

    layers = data["layers"]
    rep_layer = str(40 if 40 in layers else layers[len(layers) // 2])

    overlaps = {}
    for concept in CONCEPT_WORDS:
        proj = set(data["token_sets"].get(concept, {}).get(rep_layer, {}).get(str(K_PRIMARY), []))
        cos = set(data["token_sets_cosine"].get(concept, {}).get(rep_layer, [])[:K_PRIMARY])
        if proj and cos:
            overlap = len(proj & cos) / K_PRIMARY
            overlaps[concept] = overlap

    return {
        "layer": int(rep_layer),
        "k": K_PRIMARY,
        "per_concept_overlap": overlaps,
        "mean_overlap": float(np.mean(list(overlaps.values()))) if overlaps else 0,
        "min_overlap": float(min(overlaps.values())) if overlaps else 0,
        "max_overlap": float(max(overlaps.values())) if overlaps else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Full Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def run_full_analysis(layer: int):
    """Run complete analysis on sweep results at the given layer."""
    main_path = RESULTS_DIR / f"main_sweep_layer{layer}.jsonl"
    random_path = RESULTS_DIR / f"random_sweep_layer{layer}.jsonl"
    layer_path = RESULTS_DIR / "layer_sweep.jsonl"

    output = {}

    # Layer selection
    if layer_path.exists():
        log.info("Analyzing layer sweep...")
        output["layer_selection"] = analyze_layer_sweep(layer_path)
        for inj, best in output["layer_selection"]["best"].items():
            log.info("  Best layer (%s): %d (mean steerability=%.6f)",
                     inj, best["layer"], best["mean_steerability"])

    # Token set quality
    output["token_set_quality"] = token_set_quality_report()

    if not main_path.exists():
        log.error("Main sweep not found: %s", main_path)
        return output

    main_results = load_jsonl(main_path)
    log.info("Loaded %d main sweep results", len(main_results))

    random_results = None
    if random_path.exists():
        random_results = load_jsonl(random_path)
        log.info("Loaded %d random sweep results", len(random_results))

    for injection in INJECTION_CONDITIONS:
        prefix = injection
        log.info("Analyzing injection condition: %s", injection)

        # Steerability
        steer = compute_steerability(main_results, injection)
        p_values = [steer[c]["p_value"] for c in CONCEPT_WORDS]
        significant = benjamini_hochberg(p_values)
        for i, c in enumerate(CONCEPT_WORDS):
            steer[c]["bh_significant"] = significant[i]
        output[f"steerability_{prefix}"] = steer

        n_sig = sum(
            1 for c in CONCEPT_WORDS
            if steer[c].get("bh_significant") and steer[c]["steerability"] > 0
        )
        mean_s = np.mean([steer[c]["steerability"] for c in CONCEPT_WORDS])
        log.info("  Significant (BH, slope>0): %d/%d, mean=%.6f", n_sig, len(CONCEPT_WORDS), mean_s)

        # Cross-concept specificity
        spec_steered = compute_specificity_matrix(main_results, injection)
        spec_baseline = compute_specificity_matrix(main_results, injection, alpha=0)
        output[f"specificity_matrix_{prefix}"] = spec_steered
        output[f"specificity_matrix_normalized_{prefix}"] = row_normalize_matrix(spec_steered)
        output[f"specificity_baseline_{prefix}"] = spec_baseline
        output[f"specificity_dominance_{prefix}"] = specificity_diagonal_dominance(spec_steered)
        log.info("  Diagonal dominance: %s", output[f"specificity_dominance_{prefix}"])

        # Entropy
        output[f"entropy_flags_{prefix}"] = flag_entropy_breakdown(main_results, injection)
        log.info("  Entropy flags: %d/%d",
                 output[f"entropy_flags_{prefix}"]["n_flagged"],
                 output[f"entropy_flags_{prefix}"]["n_total"])

        # K sensitivity
        output[f"k_sensitivity_{prefix}"] = k_sensitivity_analysis(main_results, injection)

        # Abstract vs concrete
        output[f"abstract_vs_concrete_{prefix}"] = abstract_vs_concrete(steer)
        avc = output[f"abstract_vs_concrete_{prefix}"]
        log.info("  Abstract mean=%.6f, Concrete mean=%.6f",
                 avc["abstract"]["mean"], avc["concrete"]["mean"])

        # Real vs random
        if random_results:
            output[f"real_vs_random_{prefix}"] = compare_real_vs_random(
                main_results, random_results, injection
            )
            log.info("  Real vs random: %d/%d exceed 95th percentile",
                     output[f"real_vs_random_{prefix}"]["n_exceeding_95th"],
                     len(CONCEPT_WORDS))

    # Prompt validation
    output["prompt_validation"] = validate_prompts(layer)

    # Steerability with entropy-breakdown points removed
    for injection in INJECTION_CONDITIONS:
        flags = output[f"entropy_flags_{injection}"]
        if flags["n_flagged"] > 0:
            flagged_keys = {
                (f["concept"], f["prompt_idx"], f["alpha"])
                for f in flags.get("flagged_samples", [])
            }
            clean_results = [
                r for r in main_results
                if (r["concept"], r["prompt_idx"], r["alpha"]) not in flagged_keys
            ]
            output[f"steerability_clean_{injection}"] = compute_steerability(
                clean_results, injection
            )

    # Save
    analysis_path = RESULTS_DIR / f"analysis_layer{layer}.json"
    with open(analysis_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_convert)
    log.info("Analysis saved to %s", analysis_path)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"LOGIT-SHIFT STEERING VALIDATION — LAYER {layer}")
    print("=" * 80)

    for injection in INJECTION_CONDITIONS:
        steer = output[f"steerability_{injection}"]
        concepts_sorted = sorted(CONCEPT_WORDS, key=lambda c: steer[c]["steerability"], reverse=True)

        n_sig = sum(1 for c in CONCEPT_WORDS if steer[c].get("bh_significant") and steer[c]["steerability"] > 0)
        print(f"\n--- {injection} ---")
        print(f"Significant: {n_sig}/{len(CONCEPT_WORDS)}")
        print(f"{'Concept':<20} {'Steerability':>12} {'p-value':>10} {'Sig':>5} {'Category':>10}")
        print("-" * 60)
        for c in concepts_sorted:
            s = steer[c]
            sig = "YES" if s.get("bh_significant") and s["steerability"] > 0 else ""
            cat = "abstract" if c in ABSTRACT_CONCEPTS else "concrete"
            print(f"{c:<20} {s['steerability']:>12.6f} {s['p_value']:>10.4f} {sig:>5} {cat:>10}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Logit-Shift Analysis")
    parser.add_argument("--layer", type=int, help="Run full analysis at this layer")
    parser.add_argument("--layer-sweep", action="store_true", help="Analyze layer sweep only")
    parser.add_argument("--validate-prompts", action="store_true")
    args = parser.parse_args()

    if args.layer_sweep:
        path = RESULTS_DIR / "layer_sweep.jsonl"
        if not path.exists():
            print(f"Layer sweep not found: {path}")
            return
        result = analyze_layer_sweep(path)
        print(json.dumps(result, indent=2, default=_json_convert))

    elif args.validate_prompts and args.layer:
        result = validate_prompts(args.layer)
        print(json.dumps(result, indent=2, default=_json_convert))

    elif args.layer:
        run_full_analysis(args.layer)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
