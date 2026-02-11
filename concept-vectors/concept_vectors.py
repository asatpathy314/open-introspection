"""Compute and cache concept direction vectors per layer."""

import os
from pathlib import Path
from typing import Callable

import torch
from dotenv import load_dotenv
from nnsight import CONFIG, LanguageModel

from prompts import build_concept_prompt_messages, tokenize_concept_prompt

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


def _model_key(model: LanguageModel | None) -> str | None:
    if model is None:
        return None
    if hasattr(model, "to_model_key"):
        try:
            return model.to_model_key()
        except Exception:  # noqa: BLE001
            pass
    return getattr(model, "repo_id", None)


def _layer_envoy(model: LanguageModel, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError("Could not locate transformer layers on this LanguageModel.")


def _resolve_layer_envoys(model: LanguageModel, layers: list[int]):
    return [_layer_envoy(model, layer_idx) for layer_idx in layers]


def _collect_last_token_activations(
    model: LanguageModel,
    prompt_messages: list[list[dict[str, str]]],
    layers: list[int],
    remote: bool,
    logger: Callable[[str], None] | None = None,
) -> list[dict[int, torch.Tensor]]:
    """
    Collect per-prompt, per-layer residual activations at the final prefill token.

    Returns:
      {
        prompt_idx: {
          layer_idx: tensor[hidden_dim]
        }
      }
    """
    if getattr(model, "tokenizer", None) is None:
        raise RuntimeError("LanguageModel tokenizer is not initialized.")

    layer_envoys = _resolve_layer_envoys(model, layers)
    activations: list[dict[int, torch.Tensor]] = [{} for _ in prompt_messages]
    for prompt_idx, messages in enumerate(prompt_messages):
        if logger:
            logger(f"Tracing prompt {prompt_idx + 1}/{len(prompt_messages)}")

        tokenized_prompt = tokenize_concept_prompt(model.tokenizer, messages)
        if type(tokenized_prompt).__module__.startswith(
            "transformers.tokenization_utils_base"
        ):
            raise RuntimeError(
                "Prompt payload is a transformers BatchEncoding. NDIF remote tracing "
                "requires plain dict/tensor inputs. Convert tokenized prompts to a "
                "plain dict before tracer.invoke(...)."
            )

        prompt_tensor = None
        with model.trace(remote=remote) as tracer:
            with tracer.invoke(tokenized_prompt):
                layer_rows = []
                for layer_envoy in layer_envoys:
                    layer_hidden = layer_envoy.output[0]
                    # Works for both [batch, seq, hidden] and [seq, hidden].
                    layer_rows.append(
                        layer_hidden[..., -1, :].squeeze(0).detach().cpu()
                    )
                prompt_tensor = torch.stack(layer_rows, dim=0).save()

        if prompt_tensor is None:
            raise RuntimeError(
                f"Missing activations after trace. prompt_idx={prompt_idx}, no tensor returned."
            )
        if prompt_tensor.ndim != 2:
            raise RuntimeError(
                f"Unexpected prompt activation shape for prompt_idx={prompt_idx}: "
                f"{tuple(prompt_tensor.shape)} (expected 2D: layers x hidden)."
            )
        if prompt_tensor.shape[0] != len(layers):
            raise RuntimeError(
                f"Layer count mismatch for prompt_idx={prompt_idx}: "
                f"{prompt_tensor.shape[0]} vs expected {len(layers)}."
            )

        for layer_pos, layer_idx in enumerate(layers):
            activations[prompt_idx][layer_idx] = prompt_tensor[layer_pos]

    return activations


def _baseline_cache_matches(
    payload: dict,
    baseline_words: list[str] | None,
    layers: list[int] | None,
    model_key: str | None,
) -> bool:
    if "baseline_mean" not in payload:
        return False
    if baseline_words is not None and payload.get("baseline_words") != baseline_words:
        return False
    if layers is not None and payload.get("layers") != layers:
        return False
    if model_key is not None and payload.get("model_key") not in (None, model_key):
        return False
    return True


def compute_baseline_activations(
    model: LanguageModel | None = None,
    baseline_words: list[str] | None = None,
    layers: list[int] | None = None,
    remote: bool = True,
    path: str | Path = _BASELINE_CACHE_PATH,
    logger: Callable[[str], None] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Load baseline activations from cache when metadata matches; otherwise compute
    and cache fresh baseline means.
    """
    cache_path = Path(path)
    layer_indices = _normalize_layers(layers) if layers is not None else None
    model_key = _model_key(model)

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            if _baseline_cache_matches(payload, baseline_words, layer_indices, model_key):
                return payload["baseline_mean"]
        elif baseline_words is None and layer_indices is None and model_key is None:
            return payload

    if model is None or baseline_words is None or layer_indices is None:
        raise ValueError(
            "Cache miss or metadata mismatch for baseline activations. "
            "Provide model, baseline_words, and layers."
        )
    if not baseline_words:
        raise ValueError("`baseline_words` must be non-empty.")

    _configure_ndif(remote=remote)
    baseline_prompts = [build_concept_prompt_messages(word) for word in baseline_words]
    baseline_activations = _collect_last_token_activations(
        model=model,
        prompt_messages=baseline_prompts,
        layers=layer_indices,
        remote=remote,
        logger=logger,
    )

    baseline_mean: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        layer_samples = [prompt[layer_idx] for prompt in baseline_activations]
        baseline_mean[layer_idx] = torch.stack(layer_samples, dim=0).mean(dim=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline_words": baseline_words,
            "layers": layer_indices,
            "model_key": model_key,
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
    logger: Callable[[str], None] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    For each layer and word, extract residual stream activation at the final
    prefill token of the chat-formatted concept prompt and subtract mean baseline.

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
    model_key = _model_key(model)

    if _CONCEPT_CACHE_PATH.exists():
        payload = torch.load(_CONCEPT_CACHE_PATH, map_location="cpu", weights_only=True)
        if (
            isinstance(payload, dict)
            and "concept_vectors" in payload
            and payload.get("words") == words
            and payload.get("baseline_words") == baseline_words
            and payload.get("layers") == layer_indices
            and payload.get("model_key") in (None, model_key)
        ):
            return payload["concept_vectors"]

    baseline_mean = compute_baseline_activations(
        model=model,
        baseline_words=baseline_words,
        layers=layer_indices,
        remote=remote,
        logger=logger,
    )

    concept_prompts = [build_concept_prompt_messages(word) for word in words]
    activations = _collect_last_token_activations(
        model=model,
        prompt_messages=concept_prompts,
        layers=layer_indices,
        remote=remote,
        logger=logger,
    )

    if logger:
        logger(f"Post-processing {len(activations)} concept prompts")

    concept_vectors: dict[int, dict[str, torch.Tensor]] = {
        layer_idx: {} for layer_idx in layer_indices
    }
    for layer_idx in layer_indices:
        for word_idx, word in enumerate(words):
            concept_vectors[layer_idx][word] = (
                activations[word_idx][layer_idx] - baseline_mean[layer_idx]
            )

    _CONCEPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "words": words,
            "baseline_words": baseline_words,
            "layers": layer_indices,
            "model_key": model_key,
            "concept_vectors": concept_vectors,
            "baseline_mean": baseline_mean,
        },
        _CONCEPT_CACHE_PATH,
    )

    return concept_vectors


def load_concept_vectors(path: str) -> dict[int, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "concept_vectors" in payload:
        return payload["concept_vectors"]
    return payload
