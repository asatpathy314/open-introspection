#!/usr/bin/env python3
"""Analyze experiment results and print summary statistics.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --experiment self_report
    python scripts/analyze_results.py --results-dir data/results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from introspection.evaluation.results import ResultStore


def print_summary(store: ResultStore, experiment: str) -> None:
    """Print formatted summary for an experiment."""
    summary = store.summary(experiment)

    if summary["total_trials"] == 0:
        print(f"\n  {experiment}: No results found.")
        return

    print(f"\n  {experiment}:")
    print(f"    Total trials: {summary['total_trials']}")
    print(f"    Overall success rate: {summary['success_rate']:.1%}")

    if "by_layer" in summary:
        print("    By layer:")
        for layer, rate in sorted(summary["by_layer"].items()):
            print(f"      Layer {layer}: {rate:.1%}")

    if "by_alpha" in summary:
        print("    By alpha:")
        for alpha, rate in sorted(summary["by_alpha"].items()):
            print(f"      alpha={alpha}: {rate:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Results directory",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to analyze (default: all)",
    )
    args = parser.parse_args()

    store = ResultStore(args.results_dir)

    experiments = ["self_report", "distinguish", "prefill_detect", "intentional_control"]
    if args.experiment:
        experiments = [args.experiment]

    print("=" * 60)
    print("Introspection Replication - Results Summary")
    print("=" * 60)

    for exp in experiments:
        print_summary(store, exp)

    print()


if __name__ == "__main__":
    main()
