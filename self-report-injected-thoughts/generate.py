"""Generation functions for the injected-thoughts self-report experiment.

Exports two functions:
  - generate_with_injection: steering vector injected at a specific layer
  - generate_control: plain generation with no intervention

Both return the full sequence of generated token IDs via tracer.result.save().

Key assumptions:
  - model is already loaded and the NDIF API key is configured.
  - These functions must be called from the __main__ module due to nnsight's
    source-code analysis constraint (model.generate contexts only work when the
    calling code is in the top-level script).
"""

from __future__ import annotations

from dataclasses import dataclass

import nnsight
import torch


@dataclass
class TrialConfig:
    """Parameters for a single generation trial.

    Attributes:
        layer_idx: Layer index at which to inject the concept vector.
        alpha: Injection scale factor. Ignored by generate_control.
        inject_start_idx: Token position from which injection begins (prefill).
        max_new_tokens: Maximum tokens to generate.
        do_sample: Whether to use sampling; if False, uses greedy decoding.
        temperature: Sampling temperature.
        remote: Whether to route the forward pass through NDIF.
    """

    layer_idx: int
    alpha: float
    inject_start_idx: int
    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 1.0
    remote: bool = True


def generate_with_injection(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    vector: torch.Tensor,
    config: TrialConfig,
) -> torch.Tensor:
    """Generate text with a concept vector injected at a specific layer.

    Purpose: During the prefill step, adds alpha * vector to all hidden states
    at config.layer_idx from inject_start_idx onward. During autoregressive
    decoding, adds the same scaled vector at every new token position.

    Assumptions:
        - model is loaded and NDIF API key is configured.
        - input_ids has shape [1, seq_len].
        - vector has shape [hidden_dim] matching the model's hidden size.

    Args:
        model: The nnsight language model.
        input_ids: Token IDs, shape [1, seq_len].
        vector: Concept vector to inject, shape [hidden_dim].
        config: Trial configuration.

    Returns:
        torch.Tensor: Full generated token IDs (prompt + response).

    Side effects:
        Makes an NDIF network call if config.remote=True.
    """
    with model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        remote=config.remote,
    ) as tracer:
        # Prefill: inject from inject_start_idx onward
        hs = model.model.layers[config.layer_idx].output[0]  # (seq_len, hidden)
        intervention = torch.zeros_like(hs)
        intervention[config.inject_start_idx :, :] = vector * config.alpha
        model.model.layers[config.layer_idx].output[0] = hs + intervention
        # Scale vector once onto the device; reused for all autoregressive steps
        scaled = config.alpha * vector.to(device=hs.device, dtype=hs.dtype)

        # Autoregressive decoding: inject on every new token
        for _ in tracer.iter[:]:
            hs = model.model.layers[config.layer_idx].output[0]  # (1, hidden)
            model.model.layers[config.layer_idx].output[0] = hs + scaled

        output = tracer.result.save()

    return output


def generate_control(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    config: TrialConfig,
) -> torch.Tensor:
    """Generate text with no intervention (baseline / control trial).

    Purpose: Produce a response to the trial prompt without any concept injection,
    providing a baseline distribution against which injection trials are compared.

    Assumptions:
        - model is loaded and NDIF API key is configured.
        - input_ids has shape [1, seq_len].
        - Must be called from the __main__ module (nnsight source-analysis constraint).
        - config.layer_idx and config.alpha are unused.

    Args:
        model: The nnsight language model.
        input_ids: Token IDs, shape [1, seq_len].
        config: Trial configuration (only generation params are used).

    Returns:
        torch.Tensor: Full generated token IDs (prompt + response).

    Side effects:
        Makes an NDIF network call if config.remote=True.
    """
    with model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        remote=config.remote,
    ) as tracer:
        output = tracer.result.save()
    return output
