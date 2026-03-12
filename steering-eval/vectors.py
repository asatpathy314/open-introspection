"""Load and normalize concept vectors."""

import torch
from pathlib import Path
from config import VECTOR_DIR, CONCEPT_WORDS


def load_concept_vector(concept: str, vector_dir: Path = VECTOR_DIR) -> torch.Tensor:
    """Load a concept vector [num_layers, hidden_dim] from disk."""
    slug = concept.lower().replace(" ", "_")
    path = vector_dir / f"{slug}_all_layers.pt"
    if not path.exists():
        raise FileNotFoundError(f"Vector not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True).float()


def load_baseline_mean(vector_dir: Path = VECTOR_DIR) -> torch.Tensor:
    """Load baseline mean [num_layers, hidden_dim]."""
    path = vector_dir / "baseline_mean.pt"
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True).float()


def load_all_concept_vectors(
    concepts: list[str] | None = None,
    vector_dir: Path = VECTOR_DIR,
) -> dict[str, torch.Tensor]:
    """Load all concept vectors into a dict."""
    concepts = concepts or CONCEPT_WORDS
    return {c: load_concept_vector(c, vector_dir) for c in concepts}


def normalize_vector(
    vec: torch.Tensor,
    method: str,
    baseline_norm: float | None = None,
) -> torch.Tensor:
    """
    Normalize a single-layer concept vector [hidden_dim].

    Methods:
      - "raw": no normalization (Lindsey default)
      - "unit": unit norm
      - "norm_matched": scaled to match mean residual stream norm (Tan et al.)
    """
    if method == "raw":
        return vec
    norm = vec.norm()
    if norm < 1e-10:
        return vec
    unit = vec / norm
    if method == "unit":
        return unit
    if method == "norm_matched":
        if baseline_norm is None:
            raise ValueError("baseline_norm required for norm_matched normalization")
        return unit * baseline_norm
    raise ValueError(f"Unknown normalization method: {method}")


def get_baseline_norms(vector_dir: Path = VECTOR_DIR) -> torch.Tensor:
    """Return per-layer L2 norms of the baseline mean [num_layers]."""
    bl = load_baseline_mean(vector_dir)
    return bl.norm(dim=1)
