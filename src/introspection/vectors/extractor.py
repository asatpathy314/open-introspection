"""Concept vector extraction via contrastive activation analysis.

Implements the core methodology from Lindsey (2025):
    v_concept = h("Tell me about {concept}.") - mean(h("Tell me about {baseline}."))
    v_concept = v_concept / ||v_concept||_2

Activations are extracted at the residual stream output of a specified layer,
at the last token position of the input prompt.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from tqdm import tqdm

from introspection.hardware.memory import MemoryTracker
from introspection.vectors.cache import VectorCache

if TYPE_CHECKING:
    from introspection.models.loader import ModelManager
    from introspection.vectors.dataset import ConceptDataset

logger = logging.getLogger(__name__)

EXTRACTION_TEMPLATE = "Tell me about {word}."


class VectorExtractor:
    """Extracts concept vectors from model activations using contrastive analysis."""

    def __init__(self, model_manager: ModelManager, cache: VectorCache) -> None:
        self._model = model_manager
        self._cache = cache

    @property
    def cache(self) -> VectorCache:
        return self._cache

    def extract_activation(
        self,
        text: str,
        layer: int,
        token_position: Literal["last", "average"] = "last",
    ) -> torch.Tensor:
        """Run a single forward pass and extract hidden state at the specified layer.

        Args:
            text: Input text to process.
            layer: Layer index to extract from.
            token_position: 'last' for final token, 'average' for mean across sequence.

        Returns:
            Tensor of shape [hidden_dim] on CPU.
        """
        model = self._model.load()
        layer_module = self._model.get_layer_output(layer)

        with torch.no_grad():
            with model.trace(text) as tracer:
                hidden = layer_module.output[0]
                if token_position == "last":
                    vec = hidden[0, -1, :].save()
                else:
                    vec = hidden[0, :, :].mean(dim=0).save()

        return vec.value.detach().cpu().float()

    def compute_concept_vector(
        self,
        concept_word: str,
        baselines: list[str],
        layer: int,
        token_position: Literal["last", "average"] = "last",
    ) -> torch.Tensor:
        """Compute a concept vector using contrastive activation analysis.

        Formula: v = normalize(h_concept - mean(h_baselines))

        Uses Welford's running mean to avoid storing all baseline vectors
        simultaneously (constant memory regardless of baseline count).

        Args:
            concept_word: The target concept word.
            baselines: List of baseline words for the contrastive subtraction.
            layer: Layer index to extract from.
            token_position: Token position strategy.

        Returns:
            L2-normalized concept vector of shape [hidden_dim] on CPU.
        """
        # Check cache first
        cached = self._cache.get(concept_word, layer, token_position)
        if cached is not None:
            logger.debug("Using cached vector for '%s' at layer %d", concept_word, layer)
            return cached

        logger.info("Computing concept vector for '%s' at layer %d", concept_word, layer)

        # Extract concept activation
        concept_text = EXTRACTION_TEMPLATE.format(word=concept_word)
        concept_h = self.extract_activation(concept_text, layer, token_position)
        MemoryTracker.force_cleanup()

        # Compute baseline mean using Welford's running mean (constant memory)
        hidden_dim = concept_h.shape[0]
        running_mean = torch.zeros(hidden_dim, dtype=torch.float32)

        for i, baseline_word in enumerate(baselines):
            baseline_text = EXTRACTION_TEMPLATE.format(word=baseline_word)
            baseline_h = self.extract_activation(baseline_text, layer, token_position)
            # Welford's online mean: mean_new = mean_old + (x - mean_old) / (i + 1)
            running_mean += (baseline_h - running_mean) / (i + 1)
            MemoryTracker.force_cleanup()

        # Contrastive subtraction and L2 normalization
        concept_vec = concept_h - running_mean
        norm = concept_vec.norm()
        if norm > 0:
            concept_vec = concept_vec / norm
        else:
            logger.warning("Zero-norm concept vector for '%s' at layer %d", concept_word, layer)

        # Cache to disk
        self._cache.put(concept_word, layer, token_position, concept_vec)

        return concept_vec

    def extract_all_vectors(
        self,
        dataset: ConceptDataset,
        layers: list[int],
        token_position: Literal["last", "average"] = "last",
    ) -> None:
        """Pre-compute and cache all concept vectors for all layers.

        Skips concepts that are already cached. This is the batch preparation
        step run once before experiments begin.
        """
        total = len(dataset.concepts) * len(layers)
        done = 0
        skipped = 0

        for layer in layers:
            for concept in dataset.concepts:
                if self._cache.has(concept.word, layer, token_position):
                    skipped += 1
                    done += 1
                    continue

                self.compute_concept_vector(
                    concept.word,
                    dataset.baselines,
                    layer,
                    token_position,
                )
                done += 1
                logger.info(
                    "Progress: %d/%d vectors computed (%d skipped/cached)",
                    done,
                    total,
                    skipped,
                )

        logger.info(
            "Vector extraction complete: %d computed, %d from cache",
            total - skipped,
            skipped,
        )
