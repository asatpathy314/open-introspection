#!/usr/bin/env python3
"""Pre-compute all concept vectors.

This script extracts and caches concept vectors for all concepts across
all configured layers. Run this once before starting experiments to avoid
redundant computation during experiment runs.

Usage:
    python scripts/extract_vectors.py
    python scripts/extract_vectors.py --config config/default.yaml
    python scripts/extract_vectors.py --model llama-8b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from introspection.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute concept vectors")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name",
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.config, model_override=args.model)
    runner.run_vector_extraction()


if __name__ == "__main__":
    main()
