"""Memory tracking and safety utilities.

Provides monitoring, logging, and guard context managers to prevent OOM.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Generator

import psutil
import torch

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Utility for monitoring GPU/unified/system memory."""

    @staticmethod
    def snapshot() -> dict:
        """Return current memory usage stats in GB."""
        stats: dict = {"system_used_gb": 0.0, "system_total_gb": 0.0}

        vm = psutil.virtual_memory()
        stats["system_used_gb"] = round(vm.used / (1024**3), 2)
        stats["system_total_gb"] = round(vm.total / (1024**3), 2)

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                stats[f"cuda:{i}_allocated_gb"] = round(allocated, 2)
                stats[f"cuda:{i}_reserved_gb"] = round(reserved, 2)
                stats[f"cuda:{i}_total_gb"] = round(total, 2)
        elif torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                stats["mps_allocated_gb"] = round(allocated, 2)
            except Exception:
                stats["mps_allocated_gb"] = -1.0

        return stats

    @staticmethod
    def log_usage(label: str) -> None:
        """Log a memory snapshot with a human-readable label."""
        stats = MemoryTracker.snapshot()
        parts = [f"{label}:"]
        for key, val in stats.items():
            if "allocated" in key or "used" in key:
                parts.append(f"  {key}={val:.2f}GB")
        logger.info(" ".join(parts))

    @staticmethod
    def available_gb() -> float:
        """Estimate available memory in GB for the primary compute device."""
        if torch.cuda.is_available():
            # Sum available across all GPUs
            total_available = 0.0
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                total_available += total - allocated
            return total_available
        elif torch.backends.mps.is_available():
            vm = psutil.virtual_memory()
            return vm.available / (1024**3)
        else:
            vm = psutil.virtual_memory()
            return vm.available / (1024**3)

    @staticmethod
    def check_headroom(required_gb: float) -> bool:
        """Return True if enough memory remains for the operation."""
        available = MemoryTracker.available_gb()
        return available >= required_gb

    @staticmethod
    def force_cleanup() -> None:
        """Aggressive garbage collection and cache clearing."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @staticmethod
    @contextmanager
    def track(label: str) -> Generator[None, None, None]:
        """Context manager: log memory before and after, with delta."""
        before = MemoryTracker.snapshot()
        MemoryTracker.log_usage(f"{label} [before]")
        try:
            yield
        finally:
            after = MemoryTracker.snapshot()
            MemoryTracker.log_usage(f"{label} [after]")
            # Log delta for allocated keys
            for key in after:
                if "allocated" in key and key in before:
                    delta = after[key] - before[key]
                    if abs(delta) > 0.01:
                        logger.info("  %s delta: %+.2fGB", key, delta)


class MemoryGuard:
    """Context manager that checks memory headroom on entry and cleans up on exit."""

    def __init__(self, required_gb: float, label: str) -> None:
        self.required_gb = required_gb
        self.label = label

    def __enter__(self) -> MemoryGuard:
        if not MemoryTracker.check_headroom(self.required_gb):
            MemoryTracker.force_cleanup()
            if not MemoryTracker.check_headroom(self.required_gb):
                available = MemoryTracker.available_gb()
                raise MemoryError(
                    f"Insufficient memory for '{self.label}': "
                    f"need {self.required_gb:.1f}GB, "
                    f"available {available:.1f}GB"
                )
        return self

    def __exit__(self, *args: object) -> None:
        MemoryTracker.force_cleanup()
