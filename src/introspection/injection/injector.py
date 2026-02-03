"""Activation steering via concept vector injection during generation.

Implements the core injection mechanism from Lindsey (2025):
    H^(l) = H^(l) + alpha * v^(l)

where v is a unit-normalized concept vector and alpha controls injection strength.
The injection is applied at every autoregressive step via nnsight's tracer.all().
"""

from __future__ import annotations

import logging
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
    """Performs concept vector injection during model generation via nnsight."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model = model_manager

    def _prepare_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Cast vector to model's dtype for injection.

        nnsight handles device placement within trace context,
        so we only need to match dtype.
        """
        target_dtype = self._model.dtype
        return vec.to(dtype=target_dtype)

    def generate_with_injection(
        self,
        prompt: str,
        injection: InjectionConfig,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """Generate text with a concept vector injected into the residual stream.

        The vector is added to the specified layer's residual stream output
        at every autoregressive generation step.

        Args:
            prompt: Fully formatted prompt string (after chat template applied).
            injection: Injection configuration (layer, alpha, vector).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            Generated response text (decoded, special tokens stripped).
        """
        model = self._model.load()
        vec = self._prepare_vector(injection.vector)
        layer_module = self._model.get_layer_output(injection.layer)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        try:
            with torch.no_grad():
                with model.generate(prompt, **gen_kwargs) as tracer:
                    if injection.apply_to == "all":
                        with tracer.all():
                            h = layer_module.output[0]
                            layer_module.output[0][:] = h + injection.alpha * vec
                    else:
                        # generated_only: skip the prefill pass (iter[0])
                        with tracer.iter[1:]:
                            h = layer_module.output[0]
                            layer_module.output[0][:] = h + injection.alpha * vec

                    output = model.generator.output.save()

            response = model.tokenizer.decode(
                output.value[0], skip_special_tokens=True
            )
        except RuntimeError as e:
            # MPS fallback: if sampling fails, retry with greedy
            if "MPS" in str(e) and do_sample:
                logger.warning("MPS sampling failed, retrying with greedy decoding")
                return self.generate_with_injection(
                    prompt, injection, max_new_tokens, temperature=0.0, do_sample=False
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
        model = self._model.load()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        try:
            with torch.no_grad():
                with model.generate(prompt, **gen_kwargs) as tracer:
                    output = model.generator.output.save()

            response = model.tokenizer.decode(
                output.value[0], skip_special_tokens=True
            )
        except RuntimeError as e:
            if "MPS" in str(e) and do_sample:
                logger.warning("MPS sampling failed, retrying with greedy decoding")
                return self.generate_clean(prompt, max_new_tokens, temperature=0.0, do_sample=False)
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
        # Construct prompt with prefill appended as if the model started responding
        full_prompt = prompt + prefill_text
        model = self._model.load()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            if injection is not None:
                vec = self._prepare_vector(injection.vector)
                layer_module = self._model.get_layer_output(injection.layer)

                with model.generate(full_prompt, **gen_kwargs) as tracer:
                    with tracer.all():
                        h = layer_module.output[0]
                        layer_module.output[0][:] = h + injection.alpha * vec
                    output = model.generator.output.save()
            else:
                with model.generate(full_prompt, **gen_kwargs) as tracer:
                    output = model.generator.output.save()

        response = model.tokenizer.decode(output.value[0], skip_special_tokens=True)
        MemoryTracker.force_cleanup()
        return response
