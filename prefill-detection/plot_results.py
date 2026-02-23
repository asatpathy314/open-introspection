"""
Analyze and plot results from the prefill detection experiment.

Usage:
    python analyze_results.py                         # default path
    python analyze_results.py --results path/to.json  # custom path
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_apology_rates(trials: list[dict]) -> dict:
    """Compute apology rate grouped by (layer, strength).

    Returns dict mapping (layer, strength) → {apology_rate, accept_rate, n, ...}
    """
    groups = defaultdict(lambda: {"apologize": 0, "accept": 0, "unclear": 0, "total": 0})

    for t in trials:
        if t["condition"] == "control":
            key = (None, 0)
        else:
            key = (t["layer"], t["strength"])
        groups[key][t["judge_label"]] += 1
        groups[key]["total"] += 1

    rates = {}
    for key, counts in groups.items():
        n = counts["total"]
        if n == 0:
            continue
        rates[key] = {
            "apology_rate": counts["apologize"] / n,
            "accept_rate": counts["accept"] / n,
            "unclear_rate": counts["unclear"] / n,
            "n": n,
            "layer": key[0],
            "strength": key[1],
        }
    return rates


def compute_per_concept_rates(trials: list[dict]) -> dict:
    """Compute apology rate per concept, grouped by condition."""
    groups = defaultdict(lambda: defaultdict(lambda: {"apologize": 0, "accept": 0, "unclear": 0, "total": 0}))

    for t in trials:
        concept = t["concept"]
        if t["condition"] == "control":
            cond = "control"
        else:
            cond = f"L{t['layer']}_S{t['strength']}"
        groups[concept][cond][t["judge_label"]] += 1
        groups[concept][cond]["total"] += 1

    return groups


def print_table(rates: dict):
    """Print a formatted table of apology rates."""
    print(f"\n{'Layer':<10} {'Strength':<10} {'Apology%':<12} {'Accept%':<12} {'Unclear%':<12} {'N':<6}")
    print("-" * 62)

    # Control first
    if (None, 0) in rates:
        r = rates[(None, 0)]
        print(f"{'control':<10} {'-':<10} {r['apology_rate']*100:<12.1f} {r['accept_rate']*100:<12.1f} {r['unclear_rate']*100:<12.1f} {r['n']:<6}")
        print("-" * 62)

    # Injection conditions sorted by layer then strength
    injection_keys = sorted(
        [k for k in rates if k[0] is not None], key=lambda x: (x[0], x[1])
    )
    for key in injection_keys:
        r = rates[key]
        print(f"{r['layer']:<10} {r['strength']:<10} {r['apology_rate']*100:<12.1f} {r['accept_rate']*100:<12.1f} {r['unclear_rate']*100:<12.1f} {r['n']:<6}")


def plot_results(rates: dict, output_dir: Path):
    """Generate matplotlib plots of apology rate by layer and strength."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get control baseline
    control_rate = rates.get((None, 0), {}).get("apology_rate", None)

    # Group by strength
    strengths = sorted(set(k[1] for k in rates if k[0] is not None))
    layers = sorted(set(k[0] for k in rates if k[0] is not None))

    # --- Plot 1: Apology rate vs layer, one line per strength ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for s in strengths:
        layer_vals = []
        rate_vals = []
        for l in layers:
            if (l, s) in rates:
                layer_vals.append(l)
                rate_vals.append(rates[(l, s)]["apology_rate"] * 100)
        if layer_vals:
            ax.plot(layer_vals, rate_vals, "o-", label=f"strength={s}", markersize=5)

    if control_rate is not None:
        ax.axhline(y=control_rate * 100, color="black", linestyle="--",
                    label=f"control ({control_rate*100:.1f}%)", alpha=0.7)

    ax.set_xlabel("Injection Layer", fontsize=12)
    ax.set_ylabel("Apology Rate (%)", fontsize=12)
    ax.set_title("Prefill Detection: Apology Rate by Layer & Injection Strength", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(output_dir / "apology_rate_by_layer.png", dpi=150)
    print(f"Saved: {output_dir / 'apology_rate_by_layer.png'}")
    plt.close()

    # --- Plot 2: Apology rate vs strength, one line per layer ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for l in layers:
        str_vals = []
        rate_vals = []
        for s in strengths:
            if (l, s) in rates:
                str_vals.append(s)
                rate_vals.append(rates[(l, s)]["apology_rate"] * 100)
        if str_vals:
            ax.plot(str_vals, rate_vals, "o-", label=f"layer={l}", markersize=5)

    if control_rate is not None:
        ax.axhline(y=control_rate * 100, color="black", linestyle="--",
                    label=f"control ({control_rate*100:.1f}%)", alpha=0.7)

    ax.set_xlabel("Injection Strength", fontsize=12)
    ax.set_ylabel("Apology Rate (%)", fontsize=12)
    ax.set_title("Prefill Detection: Apology Rate by Strength & Layer", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(output_dir / "apology_rate_by_strength.png", dpi=150)
    print(f"Saved: {output_dir / 'apology_rate_by_strength.png'}")
    plt.close()

    # --- Plot 3: Heatmap (layer × strength) ---
    if len(layers) > 1 and len(strengths) > 1:
        import numpy as np
        matrix = np.full((len(layers), len(strengths)), np.nan)
        for i, l in enumerate(layers):
            for j, s in enumerate(strengths):
                if (l, s) in rates:
                    matrix[i, j] = rates[(l, s)]["apology_rate"] * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels(strengths)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Injection Strength")
        ax.set_ylabel("Layer")
        ax.set_title("Apology Rate Heatmap (%)")
        plt.colorbar(im, ax=ax, label="Apology Rate %")

        # Annotate cells
        for i in range(len(layers)):
            for j in range(len(strengths)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        fig.savefig(output_dir / "apology_rate_heatmap.png", dpi=150)
        print(f"Saved: {output_dir / 'apology_rate_heatmap.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/prefill_detection/results.json")
    parser.add_argument("--plot-dir", default="data/results/prefill_detection/plots")
    args = parser.parse_args()

    results = load_results(args.results)
    trials = results["trials"]
    print(f"Loaded {len(trials)} trials from {args.results}")
    print(f"Config: {json.dumps(results.get('config', {}), indent=2)}")

    rates = compute_apology_rates(trials)
    print_table(rates)
    plot_results(rates, Path(args.plot_dir))

    # Key finding: does injection reduce apology rate?
    control = rates.get((None, 0))
    if control:
        print(f"\n--- KEY METRIC ---")
        print(f"Control apology rate: {control['apology_rate']*100:.1f}% (n={control['n']})")
        best_injection = None
        best_reduction = 0
        for key, r in rates.items():
            if key[0] is None:
                continue
            reduction = control["apology_rate"] - r["apology_rate"]
            if reduction > best_reduction:
                best_reduction = reduction
                best_injection = key
        if best_injection:
            r = rates[best_injection]
            print(
                f"Best injection: layer={best_injection[0]}, strength={best_injection[1]} "
                f"→ apology rate {r['apology_rate']*100:.1f}% "
                f"(Δ = {best_reduction*100:+.1f} pp)"
            )


if __name__ == "__main__":
    main()