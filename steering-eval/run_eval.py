#!/usr/bin/env python3
"""
Steering Vector Evaluation Harness — Orchestrator

Dispatches to individual level scripts (which must run as __main__ for nnsight).

Usage:
    python run_eval.py --extract          # Extract vectors for 3.1
    python run_eval.py --level1           # MCQ propensity
    python run_eval.py --level2           # Open-ended generation
    python run_eval.py --level3           # Likelihood
    python run_eval.py --plots            # Generate all plots
    python run_eval.py --report           # Generate summary report
    python run_eval.py --all              # Run everything
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CONCEPT_WORDS, FIGURES_DIR, MODEL_ID, RESULTS_DIR, VECTOR_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("steering-eval")

SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_script(script: str, extra_args: list[str] | None = None):
    """Run a level script as a subprocess (required for nnsight __main__ constraint)."""
    cmd = [PYTHON, str(SCRIPT_DIR / script)] + (extra_args or [])
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent))
    if result.returncode != 0:
        log.error("Script %s failed with exit code %d", script, result.returncode)
        sys.exit(result.returncode)


def step_extract(args):
    import os
    n_existing = len(list(VECTOR_DIR.glob("*_all_layers.pt"))) if VECTOR_DIR.exists() else 0
    if n_existing >= 50 and not args.force:
        log.info("All 50 vectors present in %s. Use --force to re-extract.", VECTOR_DIR)
        return
    log.info("Extracting vectors for %s ...", MODEL_ID)
    env_args = f"MODEL={MODEL_ID} OUTPUT_DIR={VECTOR_DIR}"
    subprocess.run(
        f"{env_args} {PYTHON} concept-vectors/compute_concept_vectors.py",
        shell=True, cwd=str(SCRIPT_DIR.parent),
    )


def step_plots(args):
    from plots import (
        load_jsonl, plot_propensity_curves, plot_layer_heatmap,
        plot_steerability_distribution, plot_per_sample_violins,
        plot_anti_steerability, plot_coherence_vs_accuracy,
    )
    norm = args.norm or "raw"

    mcq_path = RESULTS_DIR / f"level1_mcq_{norm}.jsonl"
    steer_path = RESULTS_DIR / f"level1_steerability_{norm}.json"

    if mcq_path.exists():
        results = load_jsonl(mcq_path)
        log.info("Plotting propensity curves (%d results)...", len(results))
        plot_propensity_curves(results)
        plot_layer_heatmap(results)

    if steer_path.exists():
        with open(steer_path) as f:
            steerability = json.load(f)
        plot_steerability_distribution(steerability)
        plot_per_sample_violins(steerability)
        plot_anti_steerability(steerability)

    gen_path = RESULTS_DIR / f"level2_generation_{norm}.jsonl"
    if gen_path.exists():
        gen_results = load_jsonl(gen_path)
        plot_coherence_vs_accuracy(gen_results)

    log.info("Plots saved to %s", FIGURES_DIR)


def step_report(args):
    from stats import benjamini_hochberg, steerability_significance
    import numpy as np

    norm = args.norm or "raw"
    steer_path = RESULTS_DIR / f"level1_steerability_{norm}.json"
    if not steer_path.exists():
        log.error("Run --level1 first.")
        return

    with open(steer_path) as f:
        steerability = json.load(f)

    concepts = sorted(steerability.keys())
    scores = [steerability[c]["steerability"] for c in concepts]

    p_values = []
    for c in concepts:
        slopes = steerability[c]["per_sample_steerabilities"]
        if len(slopes) >= 2:
            _, p = steerability_significance(slopes)
            p_values.append(p)
        else:
            p_values.append(1.0)

    significant = benjamini_hochberg(p_values, q=0.05)

    lines = [
        "# Steering Vector Evaluation Report",
        f"\nModel: {MODEL_ID}",
        f"Normalization: {norm}",
        f"Concepts evaluated: {len(concepts)}",
        "",
        "| Concept | Steerability | Anti-Steer % | p-value | Significant |",
        "|---------|-------------|-------------|---------|-------------|",
    ]

    steerable_count = 0
    for i, c in enumerate(concepts):
        s = steerability[c]["steerability"]
        af = steerability[c]["anti_steerable_fraction"]
        sig = "YES" if significant[i] else "no"
        if significant[i] and s > 0:
            steerable_count += 1
        lines.append(f"| {c} | {s:.4f} | {af:.1%} | {p_values[i]:.4f} | {sig} |")

    lines.extend([
        "",
        "## Summary",
        f"- Steerable (s > 0, p < 0.05 after BH): {steerable_count}/{len(concepts)}",
        f"- Mean steerability: {np.mean(scores):.4f}",
        f"- Median steerability: {np.median(scores):.4f}",
    ])

    # Add generation results if available
    gen_path = RESULTS_DIR / f"level2_generation_{norm}.jsonl"
    if gen_path.exists():
        gen_results = []
        with open(gen_path) as f:
            for line in f:
                if line.strip():
                    gen_results.append(json.loads(line))
        if gen_results:
            steered = [r for r in gen_results if r["alpha"] > 0]
            if steered:
                id_acc = np.mean([r["id_correct"] for r in steered])
                mention_rate = np.mean([r["concept_mentioned"] for r in steered])
                coherence = np.mean([r["coherent"] for r in steered])
                lines.extend([
                    "",
                    "## Generation Evaluation (steered only)",
                    f"- Identification accuracy: {id_acc:.1%} (chance=10%)",
                    f"- Concept mention rate: {mention_rate:.1%}",
                    f"- Coherence rate: {coherence:.1%}",
                ])

    # Add likelihood results if available
    delta_path = RESULTS_DIR / f"level3_deltas_{norm}.json"
    if delta_path.exists():
        with open(delta_path) as f:
            deltas = json.load(f)
        effective = 0
        for c, d in deltas.items():
            for a, vals in d.get("deltas_by_alpha", {}).items():
                if vals["delta_positive"] > 0 and vals["delta_negative"] < 0:
                    effective += 1
                    break
        lines.extend([
            "",
            "## Likelihood Evaluation",
            f"- Concepts with effective steering (delta_pos > 0, delta_neg < 0): {effective}/{len(deltas)}",
        ])

    report_text = "\n".join(lines)
    report_path = RESULTS_DIR / f"report_{norm}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    log.info("Report -> %s", report_path)
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Steering Vector Evaluation Harness")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--level1", action="store_true")
    parser.add_argument("--level2", action="store_true")
    parser.add_argument("--level3", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--norm", default="raw")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--concepts", type=str, nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Build extra args to forward
    extra = []
    if args.norm:
        extra += ["--norm", args.norm]
    if args.layers:
        extra += ["--layers"] + [str(l) for l in args.layers]
    if args.concepts:
        extra += ["--concepts"] + args.concepts

    if args.all or args.extract:
        step_extract(args)
    if args.all or args.level1:
        run_script("level1_mcq.py", extra)
    if args.all or args.level2:
        run_script("level2_generation.py", extra)
    if args.all or args.level3:
        run_script("level3_likelihood.py", extra)
    if args.all or args.plots:
        step_plots(args)
    if args.all or args.report:
        step_report(args)

    if not any([args.all, args.extract, args.level1, args.level2, args.level3, args.plots, args.report]):
        parser.print_help()


if __name__ == "__main__":
    main()
