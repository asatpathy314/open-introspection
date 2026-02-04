"""Top-level experiment orchestrator.

Loads configuration, initializes all components, and dispatches
to individual experiment classes. Can be used as CLI entry point
or imported for programmatic use.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

from introspection.evaluation.judge import LLMJudge
from introspection.evaluation.results import ResultStore
from introspection.experiments.exp1_self_report import SelfReportExperiment
from introspection.experiments.exp2_distinguish import DistinguishExperiment
from introspection.experiments.exp3_prefill import PrefillDetectionExperiment
from introspection.experiments.exp4_intentional_control import IntentionalControlExperiment
from introspection.hardware.device import DeviceManager
from introspection.injection.injector import Injector
from introspection.models.config import get_model_config
from introspection.models.loader import ModelManager
from introspection.vectors.cache import VectorCache
from introspection.vectors.dataset import ConceptDataset
from introspection.vectors.extractor import VectorExtractor

logger = logging.getLogger(__name__)

EXPERIMENT_CLASSES = {
    "self_report": SelfReportExperiment,
    "distinguish": DistinguishExperiment,
    "prefill_detect": PrefillDetectionExperiment,
    "intentional_control": IntentionalControlExperiment,
}


class ExperimentRunner:
    """Config-driven orchestrator for introspection experiments."""

    def __init__(self, config_path: str | Path, model_override: str | None = None) -> None:
        """Initialize runner from a YAML config file.

        Args:
            config_path: Path to the main config YAML.
            model_override: Override the model name from config.
        """
        load_dotenv()

        self._config = self._load_config(config_path)

        # Set seeds for reproducibility
        seed = self._config.get("experiments", {}).get("seed", 42)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self._device = DeviceManager()
        device_info = self._device.detect()

        # Model
        model_name = model_override or self._config.get("model", {}).get("name", "gemma-4b")
        model_config = get_model_config(model_name)

        # Override quantization from config if specified
        quant = self._config.get("model", {}).get("quantization", "auto")
        if quant != "auto" and quant != model_config.quantization:
            from dataclasses import replace
            model_config = replace(model_config, quantization=quant if quant != "none" else None)

        self._model_manager = ModelManager(model_config, self._device)

        # Vectors
        cache_dir = self._config.get("vectors", {}).get("cache_dir", "data/vectors")
        self._vector_cache = VectorCache(cache_dir, model_config.cache_name)
        self._vector_extractor = VectorExtractor(self._model_manager, self._vector_cache)

        # Dataset
        dataset_path = self._config.get("vectors", {}).get(
            "dataset_path", "data/concepts/simple_data.json"
        )
        self._dataset = ConceptDataset.load(dataset_path)

        # Injection
        self._injector = Injector(self._model_manager)

        # Judge
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set. LLM judge will fail. "
                "Set it in .env or environment."
            )
        eval_config = self._config.get("evaluation", {})
        self._judge = LLMJudge(
            api_key=api_key,
            model=eval_config.get("judge_model", "gpt-5-nano"),
            max_retries=eval_config.get("max_retries", 3),
            retry_delay=eval_config.get("retry_delay", 1.0),
        )

        # Results
        output_dir = self._config.get("experiments", {}).get("output_dir", "data/results")
        self._result_store = ResultStore(output_dir)

        logger.info(
            "ExperimentRunner initialized: model=%s, backend=%s, dataset=%d concepts",
            model_config.friendly_name,
            device_info.backend,
            len(self._dataset.concepts),
        )

    def _load_config(self, config_path: str | Path) -> dict:
        """Load and merge config files (default + model-specific)."""
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load model-specific config if referenced
        model_name = config.get("model", {}).get("name", "gemma-4b")
        model_config_path = config_path.parent / "models" / f"{model_name}.yaml"
        if model_config_path.exists():
            with open(model_config_path) as f:
                model_yaml = yaml.safe_load(f)
            # Merge model config into main config (model-specific overrides)
            if model_yaml and "model" in model_yaml:
                config.setdefault("model", {}).update(model_yaml["model"])

        return config

    def _setup_logging(self) -> None:
        """Configure logging from config."""
        log_config = self._config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Also log to file if configured
        log_file = log_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
            logging.getLogger().addHandler(file_handler)

    def run_vector_extraction(self) -> None:
        """Pre-compute all concept vectors. Run once before experiments."""
        token_position = self._config.get("vectors", {}).get("token_position", "last")

        # Collect all layers needed across enabled experiments
        all_layers: set[int] = set()
        exp_config = self._config.get("experiments", {})
        for exp_name, exp_cls in EXPERIMENT_CLASSES.items():
            sub_config = exp_config.get(exp_name, {})
            if sub_config.get("enabled", False):
                layers = sub_config.get("layers", "auto")
                if layers == "auto":
                    all_layers.update(self._model_manager.config.default_layers)
                else:
                    all_layers.update(layers)

        if not all_layers:
            all_layers = set(self._model_manager.config.default_layers)

        logger.info(
            "Extracting vectors for %d concepts across %d layers",
            len(self._dataset.concepts),
            len(all_layers),
        )

        self._vector_extractor.extract_all_vectors(
            self._dataset, sorted(all_layers), token_position
        )

    def run_experiment(self, experiment_name: str) -> None:
        """Run a single named experiment."""
        if experiment_name not in EXPERIMENT_CLASSES:
            raise ValueError(
                f"Unknown experiment '{experiment_name}'. "
                f"Available: {list(EXPERIMENT_CLASSES.keys())}"
            )

        exp_config = self._config.get("experiments", {})
        shared_config = {
            "seed": exp_config.get("seed", 42),
            "temperature": exp_config.get("temperature", 1.0),
            "num_trials": exp_config.get("num_trials", 50),
            "token_position": self._config.get("vectors", {}).get("token_position", "last"),
            "apply_to": self._config.get("injection", {}).get("apply_to", "all"),
        }
        # Merge experiment-specific config
        shared_config.update(exp_config.get(experiment_name, {}))

        exp_cls = EXPERIMENT_CLASSES[experiment_name]
        experiment = exp_cls(
            model_manager=self._model_manager,
            vector_extractor=self._vector_extractor,
            injector=self._injector,
            judge=self._judge,
            result_store=self._result_store,
            dataset=self._dataset,
            config=shared_config,
        )

        logger.info("Running experiment: %s", experiment_name)
        results = experiment.run_sweep()

        # Print summary
        summary = self._result_store.summary(experiment_name)
        logger.info("Experiment '%s' summary: %s", experiment_name, summary)

    def run_all(self) -> None:
        """Run all enabled experiments sequentially."""
        exp_config = self._config.get("experiments", {})

        for exp_name in EXPERIMENT_CLASSES:
            sub_config = exp_config.get(exp_name, {})
            if sub_config.get("enabled", False):
                self.run_experiment(exp_name)
            else:
                logger.info("Skipping disabled experiment: %s", exp_name)


def main() -> None:
    """CLI entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run introspection replication experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENT_CLASSES.keys()) + ["all"],
        default="all",
        help="Which experiment to run (default: all enabled)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--extract-vectors",
        action="store_true",
        help="Pre-compute concept vectors before running experiments",
    )

    args = parser.parse_args()

    runner = ExperimentRunner(args.config, model_override=args.model)

    if args.extract_vectors:
        runner.run_vector_extraction()

    if args.experiment == "all":
        runner.run_all()
    else:
        runner.run_experiment(args.experiment)


if __name__ == "__main__":
    main()
