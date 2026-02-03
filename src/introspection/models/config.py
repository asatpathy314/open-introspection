"""Model configuration dataclasses and presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a model target."""

    model_id: str
    friendly_name: str
    num_layers: int
    hidden_dim: int
    param_count_b: float
    architecture: Literal["llama", "qwen"]
    quantization: str | None = None
    max_new_tokens: int = 256
    default_layers: list[int] = field(default_factory=list)

    @property
    def cache_name(self) -> str:
        """Name used for vector cache directory, includes quantization."""
        suffix = f"-{self.quantization}" if self.quantization else ""
        return f"{self.friendly_name}{suffix}"


# Pre-defined model configurations
LLAMA_8B = ModelConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    friendly_name="llama-8b",
    num_layers=32,
    hidden_dim=4096,
    param_count_b=8.0,
    architecture="llama",
    quantization=None,
    max_new_tokens=256,
    default_layers=[6, 9, 12, 15, 18, 21, 24, 28],
)

QWEN_32B = ModelConfig(
    model_id="Qwen/Qwen2.5-32B-Instruct",
    friendly_name="qwen-32b",
    num_layers=64,
    hidden_dim=5120,
    param_count_b=32.0,
    architecture="qwen",
    quantization="int8",
    max_new_tokens=256,
    default_layers=[10, 20, 30, 40, 50, 58],
)

LLAMA_70B = ModelConfig(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    friendly_name="llama-70b",
    num_layers=80,
    hidden_dim=8192,
    param_count_b=70.0,
    architecture="llama",
    quantization="int8",
    max_new_tokens=128,
    default_layers=[13, 26, 40, 53, 66, 76],
)

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama-8b": LLAMA_8B,
    "qwen-32b": QWEN_32B,
    "llama-70b": LLAMA_70B,
}


def get_model_config(name: str) -> ModelConfig:
    """Look up a model config by friendly name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
