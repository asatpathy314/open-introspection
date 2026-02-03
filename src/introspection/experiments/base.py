"""Base experiment class with shared sweep logic.

All four experiments inherit from BaseExperiment and implement
run_single_trial(). The sweep methods handle iteration over
layers, alphas, and trial counts.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from introspection.evaluation.results import TrialResult

if TYPE_CHECKING:
    from introspection.evaluation.judge import LLMJudge
    from introspection.evaluation.results import ResultStore
    from introspection.injection.injector import Injector
    from introspection.models.loader import ModelManager
    from introspection.vectors.dataset import ConceptDataset, ConceptWord
    from introspection.vectors.extractor import VectorExtractor

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Base class for all four introspection experiments."""

    experiment_name: str = "base"

    def __init__(
        self,
        model_manager: ModelManager,
        vector_extractor: VectorExtractor,
        injector: Injector,
        judge: LLMJudge,
        result_store: ResultStore,
        dataset: ConceptDataset,
        config: dict,
    ) -> None:
        self._model = model_manager
        self._vectors = vector_extractor
        self._injector = injector
        self._judge = judge
        self._results = result_store
        self._dataset = dataset
        self._config = config
        self._rng = random.Random(config.get("seed", 42))

    @abstractmethod
    def run_single_trial(
        self,
        concept: ConceptWord,
        layer: int,
        alpha: float,
        token_position: str,
    ) -> TrialResult:
        """Run one trial: inject concept at layer with alpha, evaluate response."""

    def _get_layers(self) -> list[int]:
        """Get layers to sweep from config or model defaults."""
        layers = self._config.get("layers", "auto")
        if layers == "auto":
            return self._model.config.default_layers
        return layers

    def _get_alphas(self) -> list[float]:
        """Get injection strengths to sweep."""
        return self._config.get("alphas", [1, 2, 4, 8, 16])

    def run_sweep(self) -> list[TrialResult]:
        """Run full experiment sweep over layers, alphas, and trials.

        For each (layer, alpha) combination, runs num_trials trials
        with randomly sampled concept words.
        """
        layers = self._get_layers()
        alphas = self._get_alphas()
        num_trials = self._config.get("num_trials", 50)
        token_position = self._config.get("token_position", "last")

        results: list[TrialResult] = []
        total = len(layers) * len(alphas) * num_trials

        logger.info(
            "Starting %s sweep: %d layers x %d alphas x %d trials = %d total",
            self.experiment_name,
            len(layers),
            len(alphas),
            num_trials,
            total,
        )

        trial_count = 0
        for layer in layers:
            for alpha in alphas:
                for trial_idx in range(num_trials):
                    concept = self._dataset.sample_concept(self._rng)
                    trial_count += 1

                    try:
                        result = self.run_single_trial(
                            concept, layer, alpha, token_position
                        )
                        self._results.save_trial(result)
                        results.append(result)

                        logger.info(
                            "[%d/%d] layer=%d alpha=%.1f word='%s' success=%s",
                            trial_count,
                            total,
                            layer,
                            alpha,
                            concept.word,
                            result.success,
                        )
                    except Exception:
                        logger.exception(
                            "Trial failed: layer=%d alpha=%.1f word='%s'",
                            layer,
                            alpha,
                            concept.word,
                        )

        logger.info(
            "%s sweep complete: %d/%d trials succeeded (%.1f%%)",
            self.experiment_name,
            sum(1 for r in results if r.success),
            len(results),
            100 * sum(1 for r in results if r.success) / max(len(results), 1),
        )

        return results

    def run_layer_sweep(
        self,
        alpha: float,
        num_trials: int,
    ) -> pd.DataFrame:
        """Run trials across all layers at a fixed alpha.

        Useful for plotting success rate vs. layer depth.
        """
        layers = self._get_layers()
        token_position = self._config.get("token_position", "last")
        results: list[TrialResult] = []

        for layer in layers:
            for _ in range(num_trials):
                concept = self._dataset.sample_concept(self._rng)
                try:
                    result = self.run_single_trial(
                        concept, layer, alpha, token_position
                    )
                    self._results.save_trial(result)
                    results.append(result)
                except Exception:
                    logger.exception("Trial failed: layer=%d", layer)

        rows = [r.to_flat_dict() for r in results]
        return pd.DataFrame(rows)

    def run_alpha_sweep(
        self,
        layer: int,
        num_trials: int,
    ) -> pd.DataFrame:
        """Run trials across all alphas at a fixed layer.

        Useful for plotting success rate vs. injection strength.
        """
        alphas = self._get_alphas()
        token_position = self._config.get("token_position", "last")
        results: list[TrialResult] = []

        for alpha in alphas:
            for _ in range(num_trials):
                concept = self._dataset.sample_concept(self._rng)
                try:
                    result = self.run_single_trial(
                        concept, layer, alpha, token_position
                    )
                    self._results.save_trial(result)
                    results.append(result)
                except Exception:
                    logger.exception("Trial failed: alpha=%.1f", alpha)

        rows = [r.to_flat_dict() for r in results]
        return pd.DataFrame(rows)
