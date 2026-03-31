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
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent))

import torch
from scipy import stats

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
    TOKEN_SET_FAMILY_DEFAULT,
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


def canonical_token_family(family: str | None) -> str:
    raw = family or TOKEN_SET_FAMILY_DEFAULT
    try:
        return TOKEN_SET_FAMILY_ALIASES[raw]
    except KeyError as e:
        valid = ", ".join(sorted(set(TOKEN_SET_FAMILY_ALIASES.values())))
        raise ValueError(f"Unknown token family '{raw}'. Valid options: {valid}") from e


def _token_set_blob_for_family(data: dict, family: str) -> dict:
    family = canonical_token_family(family)
    if family == "projection":
        return data["token_sets"]
    if family == "lexical_cosine":
        return data["token_sets_lexical_cosine"]
    if family == "concept_piece_cosine":
        return data["token_sets_cosine"]
    raise ValueError(f"Unsupported token family: {family}")


def _tokens_for_k(layer_blob, k: int) -> list[int]:
    if not layer_blob:
        return []
    if isinstance(layer_blob, list):
        return layer_blob[:k]
    return layer_blob.get(str(k), [])


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    _ensure_dir()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


def _compose_image_row(image_paths: list[Path], name: str, titles: list[str] | None = None) -> Path | None:
    """Compose existing images into a single horizontal summary figure."""
    valid = [p for p in image_paths if p and p.exists()]
    if not valid:
        return None

    fig, axes = plt.subplots(1, len(valid), figsize=(9 * len(valid), 8))
    if len(valid) == 1:
        axes = [axes]

    for i, (ax, path) in enumerate(zip(axes, valid)):
        ax.imshow(plt.imread(path))
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=11)

    fig.tight_layout()
    return _save(fig, name)


def _save_combined_pdf(image_paths: list[Path], output_path: Path):
    """Save one PNG per page into a combined PDF bundle."""
    valid = [p for p in image_paths if p and p.exists()]
    if not valid:
        return

    with PdfPages(output_path) as pdf:
        for path in valid:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(plt.imread(path))
            ax.axis("off")
            ax.set_title(path.name, fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    log.info("Saved %s", output_path)


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


def _build_specificity_array(matrix: dict, concepts: list[str]) -> np.ndarray:
    n = len(concepts)
    mat = np.zeros((n, n))
    for i, inj_c in enumerate(concepts):
        for j, tgt_c in enumerate(concepts):
            mat[i, j] = matrix.get(inj_c, {}).get(tgt_c, 0)
    return mat


def _column_correct(mat: np.ndarray) -> np.ndarray:
    """Subtract column means to remove per-token-set baseline response.

    Each column j captures the average logit shift of concept j's token set
    across all injected vectors.  Subtracting that mean leaves only the
    above-average boost that a specific injected vector gives to that token set.
    """
    return mat - mat.mean(axis=0, keepdims=True)


def plot_mean_propensity_matrix(analysis: dict, injection: str, column_corrected: bool = False) -> Path:
    """Heatmap of alpha-averaged mean propensities.

    This is descriptive only. Because it averages across symmetric positive and
    negative alphas, it mainly shows token-set baseline sensitivity rather than
    directional steerability.
    """
    key = f"mean_propensity_matrix_{injection}"
    matrix = analysis.get(key, {})
    if not matrix:
        log.warning("No mean propensity matrix for %s", injection)
        return None

    concepts = [c for c in CONCEPT_WORDS if c in matrix]
    mat = _build_specificity_array(matrix, concepts)
    suffix = "column_corrected" if column_corrected else "raw"
    title_suffix = "column-corrected" if column_corrected else "raw"
    colorbar_label = "Mean propensity"
    if column_corrected:
        mat = _column_correct(mat)
        colorbar_label = "Column-corrected mean propensity"

    fig, ax = plt.subplots(figsize=(16, 14))
    vmax = np.percentile(np.abs(mat), 95)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=90, fontsize=6)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=6)
    ax.set_xlabel("Target Concept Token Set")
    ax.set_ylabel("Injected Concept Vector")
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(f"Mean Propensity Matrix ({injection}, {title_suffix})")

    return _save(fig, f"mean_propensity_matrix_{suffix}_{injection}.png")


def plot_specificity_matrix(analysis: dict, injection: str, normalized: bool = False) -> Path:
    """50x50 heatmap of cross-concept steerability slopes."""
    suffix = "normalized" if normalized else "raw"
    key = f"specificity_matrix_normalized_{injection}" if normalized else f"specificity_matrix_{injection}"
    matrix = analysis.get(key, {})
    if not matrix:
        log.warning("No specificity matrix for %s (%s)", injection, suffix)
        return None

    concepts = [c for c in CONCEPT_WORDS if c in matrix]
    mat = _build_specificity_array(matrix, concepts)
    dominance = analysis.get(f"specificity_dominance_{injection}", {})
    mean_diag = dominance.get("mean_diagonal")
    mean_off = dominance.get("mean_off_diagonal")

    fig, ax = plt.subplots(figsize=(16, 14))
    vmax = np.percentile(np.abs(mat), 95) if not normalized else 2.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=90, fontsize=6)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=6)
    ax.set_xlabel("Target Concept Token Set")
    ax.set_ylabel("Injected Concept Vector")
    plt.colorbar(im, ax=ax, label="Mean propensity slope dprop/dalpha" if not normalized else "Row-normalized slope")
    if mean_diag is not None and mean_off is not None:
        ax.set_title(
            f"Cross-Concept Specificity ({injection}, {suffix})\n"
            f"mean diagonal={mean_diag:.3f}, mean off-diagonal={mean_off:.3f}"
        )
    else:
        ax.set_title(f"Cross-Concept Specificity ({injection}, {suffix})")

    return _save(fig, f"specificity_matrix_{suffix}_{injection}.png")


def plot_specificity_matrix_column_corrected(analysis: dict, injection: str) -> Path:
    """50x50 heatmap with column means subtracted.

    Column correction removes the per-token-set baseline response: the average
    logit shift that a given concept's token set receives across *all* injected
    vectors.  What remains is the above-average boost that each specific vector
    gives to each token set, making diagonal dominance a cleaner test of
    concept specificity.
    """
    matrix = analysis.get(f"specificity_matrix_{injection}", {})
    if not matrix:
        log.warning("No specificity matrix for %s", injection)
        return None

    concepts = [c for c in CONCEPT_WORDS if c in matrix]
    mat = _build_specificity_array(matrix, concepts)
    mat_cc = _column_correct(mat)

    # Diagonal values after correction
    diag = np.diag(mat_cc)
    off_diag = mat_cc[~np.eye(len(concepts), dtype=bool)]
    mean_diag = float(np.mean(diag))
    mean_off = float(np.mean(off_diag))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: heatmap
    ax = axes[0]
    vmax = np.percentile(np.abs(mat_cc), 95)
    im = ax.imshow(mat_cc, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=90, fontsize=6)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=6)
    ax.set_xlabel("Target Concept Token Set")
    ax.set_ylabel("Injected Concept Vector")
    plt.colorbar(im, ax=ax, label="Column-corrected propensity slope")
    ax.set_title(
        f"Column-Corrected Specificity ({injection})\n"
        f"mean diagonal={mean_diag:.3f}, mean off-diagonal={mean_off:.3f}"
    )

    # Right: sorted diagonal values
    ax2 = axes[1]
    order = np.argsort(diag)
    colors = ["steelblue" if diag[i] > 0 else "salmon" for i in order]
    ax2.barh(range(len(concepts)), diag[order], color=colors, edgecolor="none", height=0.8)
    ax2.set_yticks(range(len(concepts)))
    ax2.set_yticklabels([concepts[i] for i in order], fontsize=6)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.axvline(mean_off, color="gray", lw=1, ls="--", label=f"mean off-diag={mean_off:.3f}")
    ax2.set_xlabel("Column-corrected diagonal slope")
    ax2.set_title("Per-concept self-boost\n(above average)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    return _save(fig, f"specificity_matrix_column_corrected_{injection}.png")


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
        tick_labels=["Concrete", "Abstract"],
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
# 9. Entropy-Steerability Correlation (Test 1)
# ═══════════════════════════════════════════════════════════════════════════

_CAT_COLORS = {"abstract": "#e67e22", "concrete": "#2980b9"}


def _concept_color(c: str) -> str:
    return _CAT_COLORS["abstract"] if c in ABSTRACT_CONCEPTS else _CAT_COLORS["concrete"]



def plot_entropy_steerability_correlation(main_results: list[dict], analysis: dict) -> Path:
    """Scatter of ΔH (max alpha vs baseline) vs propensity steerability per concept.

    Tests whether steerability is just a proxy for distribution spreading:
    if so, expect strong positive Spearman correlation.  A negative or near-zero
    correlation means the two effects are dissociable.

    x-axis: mean H(α=max) − H(α=0) across prompts — the entropy increase the
    vector produces at peak steering strength.
    """
    max_alpha = max(ALPHAS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, injection in zip(axes, INJECTION_CONDITIONS):
        inj_rows = [r for r in main_results if r["injection"] == injection]

        # Baseline entropy per (concept, prompt)
        baseline_ent = defaultdict(dict)
        for r in inj_rows:
            if r["alpha"] == 0:
                baseline_ent[r["concept"]][r["prompt_idx"]] = r["entropy"]

        # ΔH at max alpha per concept
        delta_h_by_cp = defaultdict(list)
        for r in inj_rows:
            if r["alpha"] == max_alpha:
                bl = baseline_ent[r["concept"]].get(r["prompt_idx"])
                if bl is not None:
                    delta_h_by_cp[r["concept"]].append(r["entropy"] - bl)
        delta_h = {c: float(np.mean(delta_h_by_cp[c])) if delta_h_by_cp[c] else 0.0
                   for c in CONCEPT_WORDS}

        # Propensity steerability from pre-computed analysis
        steer = analysis.get(f"steerability_{injection}", {})
        prop_slopes = {c: steer[c]["steerability"] for c in CONCEPT_WORDS if c in steer}

        x = np.array([delta_h[c] for c in CONCEPT_WORDS])
        y = np.array([prop_slopes.get(c, 0) for c in CONCEPT_WORDS])
        rho, p = stats.spearmanr(x, y)

        ax.scatter(x, y, c=[_concept_color(c) for c in CONCEPT_WORDS],
                   alpha=0.75, s=45, edgecolors="none")
        for c in CONCEPT_WORDS:
            ax.annotate(c, (delta_h[c], prop_slopes.get(c, 0)), fontsize=5, alpha=0.55)

        m, b = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, m * xl + b, "k--", lw=1)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel(f"ΔH at α={max_alpha}  (H_steered − H_baseline)", fontsize=10)
        ax.set_ylabel("Propensity steerability (dprop/dα)", fontsize=10)
        ax.set_title(f"{injection}\nSpearman ρ={rho:.3f}, p={p:.4f}", fontsize=10)

    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(fc=_CAT_COLORS["abstract"], label="abstract"),
                             Patch(fc=_CAT_COLORS["concrete"], label="concrete")], fontsize=8)
    fig.suptitle("Test 1: Does steerability correlate with entropy increase?", fontsize=12, y=1.01)
    fig.tight_layout()
    return _save(fig, "entropy_steerability_correlation.png")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Cosine-Independent Token Set Propensity (Test 3)
# ═══════════════════════════════════════════════════════════════════════════


def plot_independent_token_propensity(
    layer: int,
    independent_family: str = "lexical_cosine",
) -> Path:
    """Expected propensity under an independent token family vs projection.

    Projection token sets are chosen by concept_vector @ W_U, so they are
    circular — those tokens are precisely where the vector has the largest
    effect. The comparison family is defined independently of the injected
    vector and is only used for offline diagnostics.

    Linear approximation: Δlogit_i ≈ α·(v @ W_U[i]), so expected propensity
    under any token set T is mean(v @ W_U[T]) − mean(v @ W_U[T_baseline]).
    """
    independent_family = canonical_token_family(independent_family)
    W_U = torch.load(RESULTS_DIR / "unembed.pt", map_location="cpu", weights_only=True).float()

    with open(RESULTS_DIR / "token_sets.json") as f:
        ts_data = json.load(f)

    layer_str = str(layer)
    baseline_tokens = ts_data["baseline_tokens"]
    W_baseline = W_U[baseline_tokens].mean(dim=0)
    independent_blob = _token_set_blob_for_family(ts_data, independent_family)

    proj_props, ind_props, overlaps, concepts_ok = [], [], [], []
    for concept in CONCEPT_WORDS:
        proj_layer_blob = ts_data["token_sets"].get(concept, {}).get(layer_str, {})
        ind_layer_blob = independent_blob.get(concept, {}).get(layer_str, {})
        proj_tokens = _tokens_for_k(proj_layer_blob, K_PRIMARY) if proj_layer_blob else []
        ind_tokens = _tokens_for_k(ind_layer_blob, K_PRIMARY) if ind_layer_blob else []
        if not proj_tokens or not ind_tokens:
            continue

        vec_path = VECTOR_DIR / f"{concept.lower().replace(' ', '_')}_all_layers.pt"
        if not vec_path.exists():
            continue
        v = torch.load(vec_path, map_location="cpu", weights_only=True).float()[layer]

        def _prop(toks):
            return float((v @ W_U[toks].T).mean() - (v @ W_baseline))

        proj_props.append(_prop(proj_tokens))
        ind_props.append(_prop(ind_tokens))
        overlaps.append(len(set(proj_tokens) & set(ind_tokens)) / K_PRIMARY)
        concepts_ok.append(concept)

    x = np.array(proj_props)
    y = np.array(ind_props)
    rho, p = stats.spearmanr(x, y)
    ratios = y / (np.abs(x) + 1e-12)
    family_label = independent_family.replace("_", " ")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter proj vs independent expected propensity, coloured by overlap
    ax = axes[0]
    sc = ax.scatter(x, y, c=overlaps, cmap="viridis", s=50, edgecolors="none", alpha=0.85)
    for i, c in enumerate(concepts_ok):
        ax.annotate(c, (x[i], y[i]), fontsize=5, alpha=0.55)
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y=x")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Proj token set: v @ W_U[T_proj]", fontsize=9)
    ax.set_ylabel(f"{family_label}: v @ W_U[T_ind]", fontsize=9)
    ax.set_title(
        f"Layer {layer} — expected propensity\nSpearman ρ={rho:.3f}, p={p:.4f} "
        f"| same-sign: {sum(xi*yi>0 for xi,yi in zip(x,y))}/{len(x)}",
        fontsize=9,
    )
    plt.colorbar(sc, ax=ax, label="Token set Jaccard overlap")
    ax.legend(fontsize=8)

    # Right: ind/proj ratio sorted, coloured by abstract/concrete
    ax2 = axes[1]
    order = np.argsort(ratios)
    rc = [_concept_color(concepts_ok[i]) for i in order]
    ax2.barh(range(len(concepts_ok)), ratios[order], color=rc, edgecolor="none", height=0.8)
    ax2.axvline(1.0, color="black", lw=1, ls="--", label="ratio = 1")
    ax2.axvline(0.0, color="gray", lw=0.5)
    ax2.set_yticks(range(len(concepts_ok)))
    ax2.set_yticklabels([concepts_ok[i] for i in order], fontsize=6)
    ax2.set_xlabel("ind / proj expected propensity", fontsize=9)
    ax2.set_title(
        f"Signal retained in {family_label}\nmedian ratio={float(np.median(ratios)):.3f}, "
        f"mean overlap={float(np.mean(overlaps)):.2f}",
        fontsize=9,
    )
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc=_CAT_COLORS["abstract"], label="abstract"),
                        Patch(fc=_CAT_COLORS["concrete"], label="concrete"),
                        plt.Line2D([0], [0], color="k", ls="--", label="ratio=1")],
               fontsize=7)

    fig.suptitle(
        f"Test 3: {family_label} token sets (linear approx, no new passes)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, f"{independent_family}_token_propensity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Generate All Plots
# ═══════════════════════════════════════════════════════════════════════════


def generate_all_plots(layer: int, token_family: str | None = None):
    """Generate all plots from analysis results."""
    from analysis import load_jsonl

    analysis_path = RESULTS_DIR / f"analysis_layer{layer}.json"
    if not analysis_path.exists():
        log.error("Run analysis first: python analysis.py --layer %d", layer)
        return

    with open(analysis_path) as f:
        analysis = json.load(f)
    selected_family = canonical_token_family(
        token_family or analysis.get("metadata", {}).get("token_family")
    )

    main_path = RESULTS_DIR / f"main_sweep_layer{layer}.jsonl"
    main_results = load_jsonl(main_path) if main_path.exists() else []
    generated_paths = []

    # Layer selection
    if "layer_selection" in analysis:
        path = plot_layer_selection(analysis)
        if path:
            generated_paths.append(path)

    for injection in INJECTION_CONDITIONS:
        # Steerability bars
        if f"steerability_{injection}" in analysis:
            path = plot_steerability_bars(analysis, injection)
            if path:
                generated_paths.append(path)

        # Dose-response
        if main_results:
            path = plot_dose_response(main_results, injection)
            if path:
                generated_paths.append(path)

        # Mean-propensity artifacts and corrected specificity
        for path in [
            plot_mean_propensity_matrix(analysis, injection, column_corrected=False),
            plot_mean_propensity_matrix(analysis, injection, column_corrected=True),
            plot_specificity_matrix(analysis, injection, normalized=False),
            plot_specificity_matrix(analysis, injection, normalized=True),
            plot_specificity_matrix_column_corrected(analysis, injection),
        ]:
            if path:
                generated_paths.append(path)

        # Abstract vs concrete
        path = plot_abstract_vs_concrete(analysis, injection)
        if path:
            generated_paths.append(path)

        # K sensitivity
        path = plot_k_sensitivity(analysis, injection)
        if path:
            generated_paths.append(path)

        # Real vs random
        path = plot_real_vs_random(analysis, injection)
        if path:
            generated_paths.append(path)

    # Injection comparison
    path = plot_injection_comparison(analysis)
    if path:
        generated_paths.append(path)

    # Entropy confound tests
    if main_results:
        path = plot_entropy_steerability_correlation(main_results, analysis)
        if path:
            generated_paths.append(path)
    path = plot_independent_token_propensity(layer, selected_family)
    if path:
        generated_paths.append(path)

    # Refresh combined artifacts so old files do not go stale.
    summary = _compose_image_row(
        [
            FIGURES_DIR / "specificity_matrix_column_corrected_all_positions.png",
            FIGURES_DIR / "specificity_matrix_column_corrected_last_token.png",
        ],
        "specificity_column_corrected.png",
        titles=["all_positions", "last_token"],
    )
    if summary:
        generated_paths.append(summary)

    _save_combined_pdf(generated_paths, RESULTS_DIR / "figures_combined.pdf")

    log.info("All plots saved to %s", FIGURES_DIR)


def main():
    parser = argparse.ArgumentParser(description="Logit-Shift Plots")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--token-family", type=str, default=TOKEN_SET_FAMILY_DEFAULT)
    args = parser.parse_args()
    generate_all_plots(args.layer, token_family=args.token_family)


if __name__ == "__main__":
    main()
