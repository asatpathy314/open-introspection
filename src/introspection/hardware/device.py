"""Hardware detection and device management.

DeviceManager is the single source of truth for all hardware decisions.
No other module should contain platform-specific checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable snapshot of detected hardware."""

    backend: Literal["cuda", "mps", "cpu"]
    device_count: int
    total_memory_gb: float
    per_device_memory_gb: list[float] = field(default_factory=list)


class DeviceManager:
    """Singleton that detects hardware and provides device placement decisions."""

    _instance: DeviceManager | None = None
    _device_info: DeviceInfo | None = None

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def detect(self) -> DeviceInfo:
        """Probe available hardware and return DeviceInfo."""
        if self._device_info is not None:
            return self._device_info

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            per_device = []
            total = 0.0
            for i in range(device_count):
                mem_gb = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                per_device.append(round(mem_gb, 2))
                total += mem_gb
            info = DeviceInfo(
                backend="cuda",
                device_count=device_count,
                total_memory_gb=round(total, 2),
                per_device_memory_gb=per_device,
            )
        elif torch.backends.mps.is_available():
            # MPS unified memory - use psutil for total system memory
            import psutil

            total_gb = psutil.virtual_memory().total / (1024**3)
            info = DeviceInfo(
                backend="mps",
                device_count=1,
                total_memory_gb=round(total_gb, 2),
                per_device_memory_gb=[round(total_gb, 2)],
            )
        else:
            import psutil

            total_gb = psutil.virtual_memory().total / (1024**3)
            info = DeviceInfo(
                backend="cpu",
                device_count=0,
                total_memory_gb=round(total_gb, 2),
                per_device_memory_gb=[],
            )

        DeviceManager._device_info = info
        logger.info(
            "Detected hardware: backend=%s, devices=%d, total_memory=%.1fGB",
            info.backend,
            info.device_count,
            info.total_memory_gb,
        )
        return info

    def get_device_map(
        self,
        model_size_gb: float,
        quantization: str | None = None,
    ) -> str | dict:
        """Return the appropriate device_map for model loading.

        - CUDA: 'auto' (accelerate handles multi-GPU distribution)
        - MPS: 'mps' if model fits, else raise
        - CPU: 'cpu'
        """
        info = self.detect()

        if info.backend == "cuda":
            return "auto"

        if info.backend == "mps":
            # model_size_gb is already estimated by the caller
            # Reserve ~20% of memory for activations, KV cache, OS
            available = info.total_memory_gb * 0.7
            if model_size_gb > available:
                raise MemoryError(
                    f"Model requires ~{model_size_gb:.1f}GB but only "
                    f"~{available:.1f}GB available on MPS (70% of {info.total_memory_gb:.1f}GB). "
                    f"Consider using quantization."
                )
            return "mps"

        return "cpu"

    def get_dtype(self, backend: str | None = None) -> torch.dtype:
        """Return the optimal dtype for the given backend.

        - CUDA: bfloat16 (Ampere+ native support)
        - MPS: float16 (limited bfloat16 support)
        - CPU: float32
        """
        if backend is None:
            backend = self.detect().backend

        if backend == "cuda":
            return torch.bfloat16
        elif backend == "mps":
            return torch.float16
        else:
            return torch.float32

    def needs_dispatch(self) -> bool:
        """Whether nnsight needs dispatch=True for this hardware config.

        Required when using device_map='auto' on CUDA (multi-GPU or accelerate).
        """
        info = self.detect()
        return info.backend == "cuda"

    @staticmethod
    def estimate_model_memory(
        param_count_b: float,
        quantization: str | None = None,
    ) -> float:
        """Estimate model memory in GB.

        Args:
            param_count_b: Number of parameters in billions.
            quantization: None, 'int8', or 'int4'.

        Returns:
            Estimated memory in GB (weights only, add ~20% for runtime overhead).
        """
        params = param_count_b * 1e9

        if quantization == "int4":
            bytes_per_param = 0.5
        elif quantization == "int8":
            bytes_per_param = 1.0
        else:
            bytes_per_param = 2.0  # FP16/BF16

        weight_gb = (params * bytes_per_param) / (1024**3)
        # Add 20% overhead for buffers, KV cache, activations
        return weight_gb * 1.2
