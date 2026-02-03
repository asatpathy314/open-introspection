"""Model loading with nnsight, cross-platform quantization, and memory management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from introspection.hardware.device import DeviceManager
from introspection.hardware.memory import MemoryTracker
from introspection.models.config import ModelConfig

if TYPE_CHECKING:
    from nnsight import LanguageModel
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ModelManager:
    """Loads and manages a single nnsight LanguageModel instance."""

    def __init__(self, config: ModelConfig, device_manager: DeviceManager | None = None) -> None:
        self._config = config
        self._device = device_manager or DeviceManager()
        self._model: LanguageModel | None = None

    def load(self) -> LanguageModel:
        """Load model with appropriate device_map, dtype, quantization, and dispatch.

        Returns:
            nnsight LanguageModel instance ready for tracing.
        """
        if self._model is not None:
            return self._model

        from nnsight import LanguageModel

        device_info = self._device.detect()
        logger.info(
            "Loading %s on %s (quantization=%s)",
            self._config.model_id,
            device_info.backend,
            self._config.quantization,
        )

        # Determine loading parameters
        model_size_gb = DeviceManager.estimate_model_memory(
            self._config.param_count_b, self._config.quantization
        )
        device_map = self._device.get_device_map(model_size_gb, self._config.quantization)
        dtype = self._device.get_dtype(device_info.backend)
        use_dispatch = self._device.needs_dispatch()

        kwargs: dict = {
            "device_map": device_map,
            "torch_dtype": dtype,
        }

        # Add quantization config for CUDA only
        if device_info.backend == "cuda" and self._config.quantization in ("int8", "int4"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(self._config.quantization == "int8"),
                load_in_4bit=(self._config.quantization == "int4"),
            )

        if use_dispatch:
            kwargs["dispatch"] = True

        MemoryTracker.log_usage("before_model_load")

        self._model = LanguageModel(self._config.model_id, **kwargs)

        MemoryTracker.log_usage("after_model_load")
        logger.info(
            "Model loaded: %s (device_map=%s, dtype=%s, dispatch=%s)",
            self._config.friendly_name,
            device_map,
            dtype,
            use_dispatch,
        )
        return self._model

    def unload(self) -> None:
        """Delete model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            MemoryTracker.force_cleanup()
            logger.info("Model unloaded: %s", self._config.friendly_name)

    def get_layer_output(self, layer_idx: int):
        """Return the nnsight proxy for model.model.layers[layer_idx].output.

        Both Llama and Qwen use model.model.layers[N] in HuggingFace transformers.
        The output is a tuple where [0] is the hidden states tensor.
        """
        model = self.load()
        return model.model.model.layers[layer_idx]

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer from the loaded model."""
        model = self.load()
        return model.tokenizer

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """The dtype being used by the loaded model."""
        return self._device.get_dtype()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
