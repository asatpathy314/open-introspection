"""Disk-backed cache for concept vectors.

Vectors are stored as .pt files keyed by model/layer/concept/position.
This avoids expensive recomputation across experiment runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class VectorCache:
    """Disk-backed cache for concept vectors."""

    def __init__(self, cache_dir: str | Path, model_name: str) -> None:
        """Initialize cache.

        Args:
            cache_dir: Root cache directory (e.g., 'data/vectors').
            model_name: Model identifier used as subdirectory (e.g., 'llama-8b').
        """
        self._dir = Path(cache_dir) / model_name
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, concept: str, layer: int, position: str) -> Path:
        # Sanitize concept word for filename
        safe_concept = concept.lower().replace(" ", "_").replace("/", "_")
        return self._dir / f"L{layer}_{safe_concept}_{position}.pt"

    def get(self, concept: str, layer: int, position: str = "last") -> torch.Tensor | None:
        """Load a cached vector from disk.

        Returns:
            Tensor on CPU if cached, None otherwise.
        """
        path = self._path(concept, layer, position)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def put(
        self,
        concept: str,
        layer: int,
        position: str,
        vector: torch.Tensor,
    ) -> None:
        """Save a vector to disk. Always stored as float32 on CPU."""
        path = self._path(concept, layer, position)
        torch.save(vector.detach().cpu().float(), path)
        logger.debug("Cached vector: %s", path.name)

    def has(self, concept: str, layer: int, position: str = "last") -> bool:
        """Check if a vector is cached."""
        return self._path(concept, layer, position).exists()

    def list_cached(self) -> list[str]:
        """Return all cached filenames."""
        return sorted(p.name for p in self._dir.glob("*.pt"))

    @property
    def cache_dir(self) -> Path:
        return self._dir
