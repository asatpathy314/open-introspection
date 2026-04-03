"""
Contains the core experiment logic from the first experiment in Lindsey's protocol.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import nnsight
import torch
from dotenv import load_dotenv
from prompt import build_trial_prompt, find_injection_start_idx

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
VECTOR_DIR = (
    Path(__file__).resolve().parent.parent / "data/vectors/llama-3.3-70b-instruct"
)
NUM_LAYERS = 80


@dataclass
class TrialConfig:
    layer_idx: int = 40
    alpha: float = 4.0
    inject_start_idx: int = 0
    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 1.0
    remote: bool = True


def configure_ndif_api_key() -> str:
    """Load NDIF API key from environment/.env and register it with nnsight.

    Returns:
        str: The API key.
    """
    load_dotenv()
    api_key = os.environ.get("NDIF_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NDIF_API_KEY was not found. Add it to your environment or .env file."
        )
    nnsight.CONFIG.set_default_api_key(api_key)  # type: ignore
    return api_key


def load_concept_vector(vector_dir: Path, concept: str, layer: int) -> torch.Tensor:
    """Load a single layer's concept vector from disk.

    Args:
        vector_dir: Directory containing concept vector files.
        concept: The concept name (e.g. "lightning").
        layer: The layer index to extract.

    Returns:
        torch.Tensor: The concept vector, shape [hidden_dim].
    """
    path = vector_dir / f"{concept}_all_layers.pt"
    all_layers = torch.load(path, weights_only=True)  # [num_layers, hidden_dim]
    return all_layers[layer]  # [hidden_dim]


def inject_at_layer(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,  # [1, seq_len]
    vector: torch.Tensor,  # [hidden_dim]
    config: TrialConfig,
) -> torch.Tensor:
    """Inject a concept vector at a specific layer and generate text.

    During the prefill step, the vector is added to all positions from
    inject_start_idx onward. During autoregressive decoding, the vector
    is added to every generated token.

    Args:
        model: The nnsight language model.
        input_ids: Token IDs, shape [1, seq_len].
        vector: The concept vector to inject, shape [hidden_dim].
        config: Trial configuration.

    Returns:
        torch.Tensor: The full generated token IDs (prompt + response).
    """
    with model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        remote=config.remote,
    ) as tracer:
        # Prefill step: inject from inject_start_idx onward
        hs = model.model.layers[config.layer_idx].output[0]  # (seq_len, hidden)
        intervention = torch.zeros_like(hs)
        intervention[config.inject_start_idx :, :] = vector * config.alpha
        model.model.layers[config.layer_idx].output[0] = hs + intervention
        scaled = config.alpha * vector.to(
            device=hs.device, dtype=hs.dtype
        )  # save the vector once onto device. this may break on multi-gpu.

        # Autoregressive decoding: inject on every new token
        for _ in tracer.iter[:]:
            hs = model.model.layers[config.layer_idx].output[0]  # (1, hidden)
            model.model.layers[config.layer_idx].output[0] = hs + scaled

        output = tracer.result.save()

    return output


if __name__ == "__main__":
    configure_ndif_api_key()
    model = nnsight.LanguageModel(MODEL_ID)
    tokenizer = model.tokenizer

    prompt, input_ids = build_trial_prompt(tokenizer)
    inject_start_idx = find_injection_start_idx(tokenizer, input_ids)
    seq_len = input_ids.shape[1]
    print(f"Prompt: {seq_len} tokens, injection starts at token {inject_start_idx}")

    CONCEPT = "lightning"
    LAYER = 40
    ALPHA = 4

    vec = load_concept_vector(VECTOR_DIR, CONCEPT, LAYER)
    print(f"Vector: {CONCEPT} layer {LAYER}, shape {vec.shape}")

    # --- Control trial (no injection) ---
    print("\n--- Control trial ---")
    ctrl_config = TrialConfig(
        layer_idx=LAYER, alpha=0, inject_start_idx=inject_start_idx
    )
    with model.generate(
        input_ids,
        max_new_tokens=ctrl_config.max_new_tokens,
        do_sample=ctrl_config.do_sample,
        temperature=ctrl_config.temperature,
        remote=ctrl_config.remote,
    ) as tracer:
        ctrl_output = tracer.result.save()

    ctrl_response = tokenizer.decode(
        ctrl_output[0][seq_len:], skip_special_tokens=True
    ).strip()
    print(f"Response:\n{ctrl_response}")

    # --- Injection trial ---
    print(f"\n--- Injection trial (alpha={ALPHA}, concept={CONCEPT}) ---")
    inj_config = TrialConfig(
        layer_idx=LAYER, alpha=ALPHA, inject_start_idx=inject_start_idx
    )
    inj_output = inject_at_layer(model, input_ids, vec, inj_config)

    inj_response = tokenizer.decode(
        inj_output[0][seq_len:], skip_special_tokens=True
    ).strip()
    print(f"Response:\n{inj_response}")

    print("\n--- Done ---")
