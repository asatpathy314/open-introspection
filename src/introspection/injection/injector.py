"""Activation steering via concept vector injection during generation.

Implements the core injection mechanism from Lindsey (2025):
    H^(l) = H^(l) + alpha * v^(l)

where v is a unit-normalized concept vector and alpha controls injection strength.
The injection is applied at every forward pass via a PyTorch forward hook on the
target decoder layer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from introspection.hardware.memory import MemoryTracker

if TYPE_CHECKING:
    from introspection.models.loader import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class InjectionConfig:
    """Configuration for a single concept vector injection."""

    layer: int
    alpha: float  # Scaling coefficient (paper sweeps 1, 2, 4, 8, 16)
    vector: torch.Tensor  # Unit-normalized concept vector [hidden_dim]
    apply_to: Literal["all", "generated_only"] = "all"


class Injector:
    """Performs concept vector injection during model generation.

    Uses PyTorch forward hooks on the target decoder layer to inject
    the concept vector into the residual stream at each forward pass.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model = model_manager

    def _prepare_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Cast vector to model's dtype for injection."""
        target_dtype = self._model.dtype
        return vec.to(dtype=target_dtype)

    @contextmanager
    def _injection_hook(self, injection: InjectionConfig):
        """Context manager that registers a forward hook for vector injection.

        The hook adds alpha * vector to the hidden states at the target layer.
        For apply_to='generated_only', skips the prefill pass (seq_len > 1).
        """
        model = self._model.load()
        hf_model = model._model
        vec = self._prepare_vector(injection.vector)

        # Get the actual PyTorch layer module
        if self._model.config.architecture == "gemma":
            layer = hf_model.model.language_model.layers[injection.layer]
        else:
            layer = hf_model.model.layers[injection.layer]

        def hook_fn(module, input, output):
            # Decoder layer output is hidden_states tensor or (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # For "generated_only", skip prefill (seq_len > 1 with KV caching)
            if injection.apply_to == "generated_only" and hidden_states.shape[1] > 1:
                return output

            v = vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
            modified = hidden_states + injection.alpha * v

            if isinstance(output, tuple):
                return (modified,) + tuple(output[1:])
            return modified

        handle = layer.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str:
        """Run HuggingFace generate and decode the response."""
        model = self._model.load()
        hf_model = model._model
        tokenizer = model.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(hf_model.device)
        attention_mask = inputs["attention_mask"].to(hf_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Decode only the generated tokens (skip input prompt)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    def generate_with_injection(
        self,
        prompt: str,
        injection: InjectionConfig,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """Generate text with a concept vector injected into the residual stream.

        The vector is added to the specified layer's hidden states output
        at every forward pass during generation.

        Args:
            prompt: Fully formatted prompt string (after chat template applied).
            injection: Injection configuration (layer, alpha, vector).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            Generated response text (decoded, special tokens stripped).
        """
        try:
            with self._injection_hook(injection):
                response = self._generate(
                    prompt, max_new_tokens, temperature, do_sample
                )
        except RuntimeError as e:
            if "MPS" in str(e) and do_sample:
                logger.warning("MPS sampling failed, retrying with greedy decoding")
                return self.generate_with_injection(
                    prompt, injection, max_new_tokens,
                    temperature=0.0, do_sample=False,
                )
            raise

        MemoryTracker.force_cleanup()
        return response

    def generate_clean(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """Generate text without any injection. Used for baselines and controls.

        Args:
            prompt: Fully formatted prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            Generated response text.
        """
        try:
            response = self._generate(
                prompt, max_new_tokens, temperature, do_sample
            )
        except RuntimeError as e:
            if "MPS" in str(e) and do_sample:
                logger.warning("MPS sampling failed, retrying with greedy decoding")
                return self.generate_clean(
                    prompt, max_new_tokens, temperature=0.0, do_sample=False
                )
            raise

        MemoryTracker.force_cleanup()
        return response

    def generate_with_prefill(
        self,
        prompt: str,
        prefill_text: str,
        injection: InjectionConfig | None = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> str:
        """Generate with a forced prefill for Experiment 3 (prefill detection).

        The model's initial response is forced to start with prefill_text,
        then it continues generating freely. Optionally inject a concept vector.

        Args:
            prompt: Fully formatted prompt string.
            prefill_text: Text to force as the beginning of the response.
            injection: Optional injection config.
            max_new_tokens: Maximum additional tokens after prefill.
            temperature: Sampling temperature.

        Returns:
            Full response text (prefill + generated continuation).
        """
        full_prompt = prompt + prefill_text
        do_sample = temperature > 0

        if injection is not None:
            with self._injection_hook(injection):
                response = self._generate(
                    full_prompt, max_new_tokens, temperature, do_sample
                )
        else:
            response = self._generate(
                full_prompt, max_new_tokens, temperature, do_sample
            )

        MemoryTracker.force_cleanup()
        return response
