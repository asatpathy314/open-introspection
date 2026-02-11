"""Compute and cache concept direction vectors per layer."""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from nnsight import CONFIG, LanguageModel

from prompts import build_concept_prompt

_ROOT_DIR = Path(__file__).resolve().parents[1]
_VECTORS_DIR = _ROOT_DIR / "data" / "vectors"
_BASELINE_CACHE_PATH = _VECTORS_DIR / "baseline" / "baseline_activations.pt"
_CONCEPT_CACHE_PATH = _VECTORS_DIR / "concept_vectors.pt"


def _configure_ndif(remote: bool) -> None:
    """Load NDIF API key from .env and register it with nnsight config."""
    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if remote and not api_key:
        raise RuntimeError(
            "NDIF_API_KEY was not found. Add it to your environment or .env file."
        )
    if api_key:
        CONFIG.set_default_api_key(api_key)


def _normalize_layers(layers: list[int]) -> list[int]:
    if not layers:
        raise ValueError("`layers` must be non-empty.")
    unique_sorted = sorted(set(layers))
    if unique_sorted[0] < 0:
        raise ValueError("`layers` must be non-negative indices.")
    return unique_sorted


def _collect_last_token_activations(
    model: LanguageModel,
    prompts: list[str],
    layers: list[int],
    remote: bool,
) -> dict[int, dict[int, torch.Tensor]]:
    """
    Collect per-prompt, per-layer residual activations at the final token.

    Returns:
      {
        prompt_idx: {
          layer_idx: tensor[hidden_dim]
        }
      }
    """
    activations: dict[int, dict[int, torch.Tensor]] = {
        prompt_idx: {} for prompt_idx in range(len(prompts))
    }

    with model.trace(remote=remote) as tracer:
        for prompt_idx, prompt in enumerate(prompts):
            with tracer.invoke(prompt):
                for layer_idx in layers:
                    last_token = (
                        model.model.layers[layer_idx].output[0][0, -1, :].detach().cpu()
                    )
                    activations[prompt_idx][layer_idx] = last_token.save()

                # Stop once the highest requested layer has run to save compute.
                tracer.stop()

    return activations


def compute_baseline_activations(
    model: LanguageModel | None = None,
    baseline_words: list[str] | None = None,
    layers: list[int] | None = None,
    remote: bool = True,
    path: str | Path = _BASELINE_CACHE_PATH,
) -> dict[int, torch.Tensor]:
    """
    Compute the baseline activations using the provided baseline words.
    Save this into the directory, and if it's already been saved in /data/vectors/baseline
    then just load and return those.
    """
    cache_path = Path(path)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "baseline_mean" in payload:
            return payload["baseline_mean"]
        return payload

    if model is None or baseline_words is None or layers is None:
        raise ValueError(
            "Cache miss for baseline activations. Provide model, baseline_words, and layers."
        )
    if not baseline_words:
        raise ValueError("`baseline_words` must be non-empty.")

    _configure_ndif(remote=remote)
    layer_indices = _normalize_layers(layers)
    baseline_prompts = [build_concept_prompt(word) for word in baseline_words]
    baseline_activations = _collect_last_token_activations(
        model=model,
        prompts=baseline_prompts,
        layers=layer_indices,
        remote=remote,
    )

    baseline_mean: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        layer_samples = [
            baseline_activations[prompt_idx][layer_idx]
            for prompt_idx in range(len(baseline_words))
        ]
        baseline_mean[layer_idx] = torch.stack(layer_samples, dim=0).mean(dim=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline_words": baseline_words,
            "layers": layer_indices,
            "baseline_mean": baseline_mean,
        },
        cache_path,
    )
    return baseline_mean


def compute_concept_vectors(
    model: LanguageModel,
    words: list[str],
    baseline_words: list[str],
    layers: list[int],
    remote: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    For each layer and word, extract residual stream activation at the last token
    of 'Tell me about {word}.' and subtract mean baseline activation.

    Returns: {layer_idx: {word: concept_vector_tensor}}
    """
    if not words:
        raise ValueError("`words` must be non-empty.")
    if not baseline_words:
        raise ValueError("`baseline_words` must be non-empty.")
    if len(words) != len(set(words)):
        raise ValueError("`words` must be unique to map cleanly into dict outputs.")

    _configure_ndif(remote=remote)
    layer_indices = _normalize_layers(layers)

    if _CONCEPT_CACHE_PATH.exists():
        payload = torch.load(_CONCEPT_CACHE_PATH, map_location="cpu", weights_only=True)
        if (
            isinstance(payload, dict)
            and "concept_vectors" in payload
            and payload.get("words") == words
            and payload.get("baseline_words") == baseline_words
            and payload.get("layers") == layer_indices
        ):
            return payload["concept_vectors"]

    # Step 1: For each word, build prompt via build_concept_prompt().
    concept_prompts = [build_concept_prompt(word) for word in words]
    baseline_prompts = [build_concept_prompt(word) for word in baseline_words]
    all_prompts = [*concept_prompts, *baseline_prompts]

    # Step 2: Batch all words + baseline_words into one remote trace and use invoke per prompt.
    activations = _collect_last_token_activations(
        model=model,
        prompts=all_prompts,
        layers=layer_indices,
        remote=remote,
    )

    # Step 3: On client, compute mean activation across baseline_words per layer.
    baseline_start = len(words)
    baseline_mean: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        baseline_layer_samples = [
            activations[baseline_start + i][layer_idx] for i in range(len(baseline_words))
        ]
        baseline_mean[layer_idx] = torch.stack(baseline_layer_samples, dim=0).mean(dim=0)

    # Step 4: Subtract mean from each word's activation to get concept direction.
    concept_vectors: dict[int, dict[str, torch.Tensor]] = {
        layer_idx: {} for layer_idx in layer_indices
    }
    for layer_idx in layer_indices:
        for word_idx, word in enumerate(words):
            concept_vectors[layer_idx][word] = (
                activations[word_idx][layer_idx] - baseline_mean[layer_idx]
            )

    # Step 5: Cache result to disk as concept_vectors.pt.
    _CONCEPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "words": words,
            "baseline_words": baseline_words,
            "layers": layer_indices,
            "concept_vectors": concept_vectors,
            "baseline_mean": baseline_mean,
        },
        _CONCEPT_CACHE_PATH,
    )

    # Also cache baseline means in dedicated baseline directory.
    _BASELINE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline_words": baseline_words,
            "layers": layer_indices,
            "baseline_mean": baseline_mean,
        },
        _BASELINE_CACHE_PATH,
    )

    return concept_vectors


def load_concept_vectors(path: str) -> dict[int, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "concept_vectors" in payload:
        return payload["concept_vectors"]
    return payload
