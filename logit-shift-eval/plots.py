#!/usr/bin/env python3
"""
Logit-Shift Steering Validation — Visualization

Usage:
    python plots.py --layer L   # Generate all plots from analysis results
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ABSTRACT_CONCEPTS,
    ALPHAS,
    CONCEPT_WORDS,
    CONCRETE_CONCEPTS,
    FIGURES_DIR,
    INJECTION_CONDITIONS,
    K_PRIMARY,
    K_VALUES,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    _ensure_dir()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. Steerability Bar Chart
# ═══════════════════════════════════════════════════════════════════════════


def plot_steerability_bars(analysis: dict, injection: str) -> Path:
    """Horizontal bar chart of per-concept steerability with significance markers."""
    steer = analysis[f"steerability_{injection}"]
    concepts = sorted(CONCEPT_WORDS, key=lambda c: steer[c]["steerability"])
    slopes = [steer[c]["steerability"] for c in concepts]
    colors = []
    for c in concepts:
        if steer[c].get("bh_significant") and steer[c]["steerability"] > 0:
            colors.append("steelblue")
        elif steer[c]["steerability"] > 0:
            colors.append("lightblue")
        elif steer[c].get("bh_significant"):
            colors.append("salmon")
        else:
            colors.append("lightsalmon")

    fig, ax = plt.subplots(figsize=(8, 14))
    ax.barh(range(len(concepts)), slopes, color=colors, edgecolor="none")
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Steerability (slope of propensity vs alpha)")
    ax.set_title(f"Per-Concept Steerability ({injection})")

    n_sig = sum(1 for c in CONCEPT_WORDS if steer[c].get("bh_significant") and steer[c]["steerability"] > 0)
    ax.text(0.02, 0.98, f"Significant: {n_sig}/{len(CONCEPT_WORDS)} (BH q=0.05)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return _save(fig, f"steerability_bars_{injection}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dose-Response Curves
# ═══════════════════════════════════════════════════════════════════════════


def plot_dose_response(main_results: list[dict], injection: str,
                       concepts: list[str] | None = None) -> Path:
    """Propensity vs alpha for selected concepts with entropy overlay."""
    filtered = [r for r in main_results if r["injection"] == injection]

    if concepts is None:
        # Auto-select: top 3 + bottom 3 by mean propensity at max alpha
        by_concept = defaultdict(list)
        for r in filtered:
            if r["alpha"] == max(ALPHAS):
                by_concept[r["concept"]].append(r[f"propensity_k{K_PRIMARY}"])
        ranked = sorted(by_concept, key=lambda c: np.mean(by_concept[c]), reverse=True)
        concepts = ranked[:3] + ranked[-3:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: propensity curves
    ax = axes[0]
    for concept in concepts:
        rows = [r for r in filtered if r["concept"] == concept]
        alpha_to_prop = defaultdict(list)
        for r in rows:
            alpha_to_prop[r["alpha"]].append(r[f"propensity_k{K_PRIMARY}"])
        alphas = sorted(alpha_to_prop.keys())
        means = [np.mean(alpha_to_prop[a]) for a in alphas]
        sems = [np.std(alpha_to_prop[a]) / max(np.sqrt(len(alpha_to_prop[a])), 1)
                for a in alphas]
        ax.errorbar(alphas, means, yerr=sems, marker="o", capsize=3, label=concept)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Alpha")
    ax.set_ylabel(f"Propensity (k={K_PRIMARY})")
    ax.set_title("Dose-Response: Propensity vs Alpha")
    ax.legend(fontsize=8)

    # Right: entropy overlay
    ax2 = axes[1]
    for concept in concepts:
        rows = [r for r in filtered if r["concept"] == concept]
        alpha_to_ent = defaultdict(list)
        for r in rows:
            alpha_to_ent[r["alpha"]].append(r["entropy"])
        alphas = sorted(alpha_to_ent.keys())
        means = [np.mean(alpha_to_ent[a]) for a in alphas]
        ax2.plot(alphas, means, marker="s", label=concept)
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Entropy")
    ax2.set_title("Output Distribution Entropy")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Dose-Response ({injection})", fontsize=13)
    plt.tight_layout()
    return _save(fig, f"dose_response_{injection}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Layer Selection
# ═══════════════════════════════════════════════════════════════════════════


def plot_layer_selection(analysis: dict) -> Path:
    """Bar chart of mean steerability by layer for each injection condition."""
    ls = analysis.get("layer_selection", {})
    scores = ls.get("layer_scores", {})
    if not scores:
        log.warning("No layer sweep data to plot")
        return None

    fig, axes = plt.subplots(1, len(INJECTION_CONDITIONS), figsize=(7 * len(INJECTION_CONDITIONS), 5))
    if len(INJECTION_CONDITIONS) == 1:
        axes = [axes]

    for ax, injection in zip(axes, INJECTION_CONDITIONS):
        layer_data = {
            v["layer"]: v["mean_steerability"]
            for v in scores.values() if v["injection"] == injection
        }
        if not layer_data:
            continue
        layers = sorted(layer_data.keys())
        vals = [layer_data[l] for l in layers]
        best_layer = layers[np.argmax(vals)]

        colors = ["steelblue" if l == best_layer else "lightgray" for l in layers]
        ax.bar(range(len(layers)), vals, color=colors, edgecolor="none")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Steerability")
        ax.set_title(f"{injection} (best: layer {best_layer})")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Layer Selection Sweep", fontsize=13)
    plt.tight_layout()
    return _save(fig, "layer_selection.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Injection Comparison Scatter
# ═══════════════════════════════════════════════════════════════════════════


def plot_injection_comparison(analysis: dict) -> Path:
    """Scatter plot of all-positions vs last-token steerability per concept."""
    s_all = analysis.get("steerability_all_positions", {})
    s_last = analysis.get("steerability_last_token", {})
    if not s_all or not s_last:
        log.warning("Need both injection conditions for comparison plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    xs, ys, labels = [], [], []
    for c in CONCEPT_WORDS:
        if c in s_all and c in s_last:
            xs.append(s_all[c]["steerability"])
            ys.append(s_last[c]["steerability"])
            labels.append(c)

    colors = ["steelblue" if c in CONCRETE_CONCEPTS else "coral" for c in labels]
    ax.scatter(xs, ys, c=colors, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Label outliers
    for i, c in enumerate(labels):
        if abs(xs[i]) > np.percentile(np.abs(xs), 90) or abs(ys[i]) > np.percentile(np.abs(ys), 90):
            ax.annotate(c, (xs[i], ys[i]), fontsize=7, alpha=0.8)

    lims = [min(min(xs), min(ys)) - 0.001, max(max(xs), max(ys)) + 0.001]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("All-Positions Steerability")
    ax.set_ylabel("Last-Token Steerability")
    ax.set_title("Injection Condition Comparison")
    ax.legend(["y=x", "Concrete", "Abstract"], fontsize=9)

    if len(xs) >= 3:
        from scipy.stats import pearsonr
        r, p = pearsonr(xs, ys)
        ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.4f}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(facecolor="wheat", alpha=0.5))

    return _save(fig, "injection_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Real vs Random
# ═══════════════════════════════════════════════════════════════════════════


def plot_real_vs_random(analysis: dict, injection: str) -> Path:
    """Histogram comparing real concept steerability vs random vectors."""
    rvr = analysis.get(f"real_vs_random_{injection}")
    if not rvr:
        log.warning("No real-vs-random data for %s", injection)
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    real_vals = [rvr["per_concept"][c]["real_steerability"] for c in CONCEPT_WORDS]
    random_all = []
    for c_data in rvr.get("all_random_steerabilities", {}).values():
        random_all.extend(c_data)

    if random_all:
        ax.hist(random_all, bins=30, alpha=0.5, color="gray", label="Random vectors", density=True)
    ax.hist(real_vals, bins=20, alpha=0.7, color="steelblue", label="Real concept vectors", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Steerability")
    ax.set_ylabel("Density")
    ax.set_title(f"Real vs Random Vector Steerability ({injection})")
    ax.legend()

    n_exc = rvr["n_exceeding_95th"]
    ax.text(0.98, 0.95,
            f"{n_exc}/{len(CONCEPT_WORDS)} concepts exceed\nrandom 95th percentile",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="wheat", alpha=0.5))

    return _save(fig, f"real_vs_random_{injection}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cross-Concept Specificity Heatmap
# ═══════════════════════════════════════════════════════════════════════════


def plot_specificity_matrix(analysis: dict, injection: str, normalized: bool = False) -> Path:
    """50x50 heatmap of cross-concept logit shifts."""
    suffix = "normalized" if normalized else "raw"
    key = f"specificity_matrix_normalized_{injection}" if normalized else f"specificity_matrix_{injection}"
    matrix = analysis.get(key, {})
    if not matrix:
        log.warning("No specificity matrix for %s (%s)", injection, suffix)
        return None

    # Build numpy matrix
    concepts = [c for c in CONCEPT_WORDS if c in matrix]
    n = len(concepts)
    mat = np.zeros((n, n))
    for i, inj_c in enumerate(concepts):
        for j, tgt_c in enumerate(concepts):
            mat[i, j] = matrix.get(inj_c, {}).get(tgt_c, 0)

    fig, ax = plt.subplots(figsize=(16, 14))
    vmax = np.percentile(np.abs(mat), 95) if not normalized else 2.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(concepts, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(concepts, fontsize=6)
    ax.set_xlabel("Target Concept Token Set")
    ax.set_ylabel("Injected Concept Vector")
    plt.colorbar(im, ax=ax, label="Mean Logit Shift" if not normalized else "Normalized Shift")
    ax.set_title(f"Cross-Concept Specificity ({injection}, {suffix})")

    return _save(fig, f"specificity_matrix_{suffix}_{injection}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Abstract vs Concrete
# ═══════════════════════════════════════════════════════════════════════════


def plot_abstract_vs_concrete(analysis: dict, injection: str) -> Path:
    """Box plot comparing abstract vs concrete concept steerability."""
    steer = analysis.get(f"steerability_{injection}", {})
    if not steer:
        return None

    abstract_vals = [steer[c]["steerability"] for c in CONCEPT_WORDS if c in ABSTRACT_CONCEPTS and c in steer]
    concrete_vals = [steer[c]["steerability"] for c in CONCEPT_WORDS if c in CONCRETE_CONCEPTS and c in steer]

    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.boxplot(
        [concrete_vals, abstract_vals],
        labels=["Concrete", "Abstract"],
        patch_artist=True,
        widths=0.5,
    )
    parts["boxes"][0].set_facecolor("steelblue")
    parts["boxes"][1].set_facecolor("coral")
    for box in parts["boxes"]:
        box.set_alpha(0.6)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Steerability")
    ax.set_title(f"Abstract vs Concrete ({injection})")

    avc = analysis.get(f"abstract_vs_concrete_{injection}", {})
    if "mann_whitney" in avc:
        p = avc["mann_whitney"]["p_value"]
        ax.text(0.98, 0.95, f"Mann-Whitney p={p:.4f}\n(H1: concrete > abstract)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(facecolor="wheat", alpha=0.5))

    return _save(fig, f"abstract_vs_concrete_{injection}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. K Sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def plot_k_sensitivity(analysis: dict, injection: str) -> Path:
    """Line plot of mean steerability at each k value."""
    ks_data = analysis.get(f"k_sensitivity_{injection}", {}).get("per_k", {})
    if not ks_data:
        return None

    ks = sorted(int(k) for k in ks_data if "mean_steerability" in ks_data.get(k, ks_data.get(str(k), {})))
    if not ks:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean steerability by k
    ax = axes[0]
    means = []
    stds = []
    for k in ks:
        d = ks_data.get(k, ks_data.get(str(k), {}))
        means.append(d["mean_steerability"])
        stds.append(d["std"])
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=5, color="steelblue")
    ax.set_xlabel("k (token set size)")
    ax.set_ylabel("Mean Steerability")
    ax.set_title("Steerability vs Token Set Size")
    ax.set_xticks(ks)

    # Right: per-concept steerability correlation between k values
    ax2 = axes[1]
    corrs = analysis.get(f"k_sensitivity_{injection}", {}).get("correlations", {})
    if corrs:
        labels = sorted(corrs.keys())
        rhos = [corrs[l]["spearman_rho"] for l in labels]
        ax2.bar(range(len(labels)), rhos, color="steelblue", alpha=0.7)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Spearman rho")
        ax2.set_title("Cross-k Consistency")
        ax2.axhline(1, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle(f"K Sensitivity ({injection})", fontsize=13)
    plt.tight_layout()
    return _save(fig, f"k_sensitivity_{injection}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Generate All Plots
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_plots(layer: int):
    """Generate all plots from analysis results."""
    from analysis import load_jsonl

    analysis_path = RESULTS_DIR / f"analysis_layer{layer}.json"
    if not analysis_path.exists():
        log.error("Run analysis first: python analysis.py --layer %d", layer)
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    main_path = RESULTS_DIR / f"main_sweep_layer{layer}.jsonl"
    main_results = load_jsonl(main_path) if main_path.exists() else []

    # Layer selection
    if "layer_selection" in analysis:
        plot_layer_selection(analysis)

    for injection in INJECTION_CONDITIONS:
        # Steerability bars
        if f"steerability_{injection}" in analysis:
            plot_steerability_bars(analysis, injection)

        # Dose-response
        if main_results:
            plot_dose_response(main_results, injection)

        # Specificity matrix (raw and normalized)
        plot_specificity_matrix(analysis, injection, normalized=False)
        plot_specificity_matrix(analysis, injection, normalized=True)

        # Abstract vs concrete
        plot_abstract_vs_concrete(analysis, injection)

        # K sensitivity
        plot_k_sensitivity(analysis, injection)

        # Real vs random
        plot_real_vs_random(analysis, injection)

    # Injection comparison
    plot_injection_comparison(analysis)

    log.info("All plots saved to %s", FIGURES_DIR)


def main():
    parser = argparse.ArgumentParser(description="Logit-Shift Plots")
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    generate_all_plots(args.layer)


if __name__ == "__main__":
    main()
