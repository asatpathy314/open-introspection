#!/usr/bin/env python3
"""CLI script to run introspection experiments.

Usage:
    # Run all enabled experiments with default config
    python scripts/run_experiment.py

    # Run specific experiment
    python scripts/run_experiment.py --experiment self_report

    # Pre-compute vectors first, then run
    python scripts/run_experiment.py --extract-vectors --experiment self_report

    # Use a specific model
    python scripts/run_experiment.py --model llama-8b --experiment self_report

    # Custom config
    python scripts/run_experiment.py --config config/default.yaml
"""

import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from introspection.experiments.runner import main

if __name__ == "__main__":
    main()
