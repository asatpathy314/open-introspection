#!/usr/bin/env python3
"""Convenience entry point. See scripts/run_experiment.py for full CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from introspection.experiments.runner import main

if __name__ == "__main__":
    main()
