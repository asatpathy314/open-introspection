"""Prompt construction for concept extraction."""

from __future__ import annotations

import torch


def build_concept_prompt_messages(word: str) -> list[dict[str, str]]:
    """Build chat messages for concept extraction."""
    return [{"role": "user", "content": f"Tell me about {word}."}]


def tokenize_concept_prompt(
    tokenizer, messages: list[dict[str, str]]
) -> dict[str, torch.Tensor]:
    """
    Apply chat template with generation prompt so the final token is at the
    assistant boundary where generation begins.

    Returns a plain dict[str, torch.Tensor] to keep NDIF payloads free of
    `transformers.tokenization_utils_base.BatchEncoding`.
    """
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # NDIF remote tracing rejects some unpickled module types, so sanitize to
    # a plain dict with tensor values only.
    payload = {key: value for key, value in tokenized.items() if torch.is_tensor(value)}
    if "input_ids" not in payload:
        raise RuntimeError(
            "Chat template tokenization did not produce `input_ids`; cannot trace prompt."
        )
    return payload
