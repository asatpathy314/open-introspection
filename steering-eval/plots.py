"""Plotting utilities for steering evaluation results."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import FIGURES_DIR


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


# ── Figure 1: Propensity curves per concept ──

def plot_propensity_curves(results: list[dict], out_path: Path | None = None):
    """Plot m_LD vs alpha for each concept, with error bars."""
    _ensure_dir()
    by_concept = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_concept[r["concept"]][r["alpha"]].append(r["propensity"])

    n_concepts = len(by_concept)
    cols = 5
    rows = (n_concepts + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, (concept, alpha_dict) in enumerate(sorted(by_concept.items())):
        ax = axes[idx // cols][idx % cols]
        alphas = sorted(alpha_dict.keys())
        means = [np.mean(alpha_dict[a]) for a in alphas]
        sems = [np.std(alpha_dict[a]) / max(np.sqrt(len(alpha_dict[a])), 1) for a in alphas]
        ax.errorbar(alphas, means, yerr=sems, marker="o", capsize=3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(concept, fontsize=9)
        ax.set_xlabel("alpha")
        ax.set_ylabel("m_LD")

    # Hide unused subplots
    for idx in range(n_concepts, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle("Propensity Curves (m_LD vs alpha)", fontsize=14)
    plt.tight_layout()
    path = out_path or FIGURES_DIR / "propensity_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Figure 2: Layer sweep heatmap ──

def plot_layer_heatmap(steerability_data: dict, out_path: Path | None = None):
    """
    Heatmap of steerability by (concept x layer).
    steerability_data: dict from level1_mcq.compute_steerability, but needs
    per-layer data. Alternatively, accepts results list and computes internally.
    """
    _ensure_dir()
    # If passed raw results, group by concept+layer and compute slopes
    if isinstance(steerability_data, list):
        by_cl = defaultdict(lambda: defaultdict(list))
        for r in steerability_data:
            by_cl[r["concept"]][r["layer"]].append(r)

        concepts = sorted(by_cl.keys())
        layers = sorted({r["layer"] for r in steerability_data})

        matrix = np.zeros((len(concepts), len(layers)))
        for ci, concept in enumerate(concepts):
            for li, layer in enumerate(layers):
                rows = by_cl[concept][layer]
                if not rows:
                    continue
                alpha_to_prop = defaultdict(list)
                for r in rows:
                    alpha_to_prop[r["alpha"]].append(r["propensity"])
                xs = sorted(alpha_to_prop.keys())
                ys = [np.mean(alpha_to_prop[a]) for a in xs]
                if len(xs) >= 2:
                    matrix[ci, li] = np.polyfit(xs, ys, 1)[0]
    else:
        raise ValueError("Pass raw results list for layer heatmap")

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.8), max(8, len(concepts) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")
    plt.colorbar(im, ax=ax, label="Steerability (slope)")
    plt.title("Steerability by Concept x Layer")
    plt.tight_layout()
    path = out_path or FIGURES_DIR / "layer_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Figure 3: Steerability distribution ──

def plot_steerability_distribution(steerability_by_concept: dict, out_path: Path | None = None):
    _ensure_dir()
    concepts = sorted(steerability_by_concept.keys())
    scores = [steerability_by_concept[c]["steerability"] for c in concepts]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if s > 0 else "red" for s in scores]
    ax.barh(concepts, scores, color=colors, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Steerability (slope)")
    ax.set_title("Per-Concept Steerability")
    plt.tight_layout()
    path = out_path or FIGURES_DIR / "steerability_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Figure 4: Per-sample steerability violin plots ──

def plot_per_sample_violins(steerability_by_concept: dict, top_n: int = 10, out_path: Path | None = None):
    _ensure_dir()
    sorted_concepts = sorted(
        steerability_by_concept.keys(),
        key=lambda c: steerability_by_concept[c]["steerability"],
        reverse=True,
    )
    selected = sorted_concepts[:top_n] + sorted_concepts[-top_n:]

    data = []
    labels = []
    for c in selected:
        slopes = steerability_by_concept[c]["per_sample_steerabilities"]
        if slopes:
            data.append(slopes)
            labels.append(c)

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Per-sample steerability")
    ax.set_title("Per-Sample Steerability: Top/Bottom Concepts")
    plt.tight_layout()
    path = out_path or FIGURES_DIR / "per_sample_violins.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Figure 5: Anti-steerability fraction bar chart ──

def plot_anti_steerability(steerability_by_concept: dict, out_path: Path | None = None):
    _ensure_dir()
    concepts = sorted(steerability_by_concept.keys())
    fracs = [steerability_by_concept[c]["anti_steerable_fraction"] for c in concepts]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(concepts)), fracs, color="salmon", alpha=0.7)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=90, fontsize=7)
    ax.set_ylabel("Anti-steerable fraction")
    ax.set_title("Fraction of Anti-Steerable Samples per Concept")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = out_path or FIGURES_DIR / "anti_steerability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Figure 6: Coherence vs identification accuracy ──

def plot_coherence_vs_accuracy(gen_results: list[dict], out_path: Path | None = None):
    _ensure_dir()
    by_config = defaultdict(lambda: {"id_correct": [], "coherent": []})
    for r in gen_results:
        key = (r["concept"], r["layer"], r["alpha"])
        by_config[key]["id_correct"].append(r["id_correct"])
        by_config[key]["coherent"].append(r["coherent"])

    accs = []
    cohs = []
    for key, data in by_config.items():
        accs.append(np.mean(data["id_correct"]))
        cohs.append(np.mean(data["coherent"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(accs, cohs, alpha=0.5)
    ax.set_xlabel("Identification Accuracy")
    ax.set_ylabel("Coherence Rate")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% coherence threshold")
    ax.axvline(0.1, color="blue", linestyle="--", alpha=0.5, label="Chance (10%)")
    ax.legend()
    ax.set_title("Coherence vs Identification Accuracy")
    plt.tight_layout()
    path = out_path or FIGURES_DIR / "coherence_vs_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
