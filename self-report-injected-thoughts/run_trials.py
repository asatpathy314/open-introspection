"""Run trials for the injected-thoughts self-report experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
VECTOR_DIR = Path("data/vectors/llama-3.3-70b-instruct")


@dataclass
class TrialConfig:
    """Configuration for a single trial of the injected-thoughts self-report experiment.

    Attributes:
        layer_idx: Index of the layer to inject the thought vector into.
        alpha: Strength of the thought vector injection.
        inject_start_idx: Position in the sequence to start injecting the thought vector.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature for sampling.
        do_sample: Whether to sample or use greedy decoding.
        remote: Whether to use NDIF API.
    """

    layer_idx: int
    alpha: float
    inject_start_idx: int
    max_new_tokens: int = 100
    temperature: float = 1.0
    do_sample: bool = True
    remote: bool = True
