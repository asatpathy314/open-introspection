"""Quick smoke test: 1 control + 1 injection trial via NDIF."""

import os
import sys
import torch
from dotenv import load_dotenv

load_dotenv()

from prompt import build_trial_prompt
from run_trials import load_concept_vector, MODEL_ID, VECTOR_DIR

# --- Setup ---
import nnsight
from nnsight import CONFIG, LanguageModel

if os.environ.get("NDIF_API_KEY"):
    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

print(f"Model: {MODEL_ID}")
print(f"Online: {nnsight.is_model_running(MODEL_ID)}")

model = LanguageModel(MODEL_ID)
tokenizer = model.tokenizer

prompt, input_ids, injection_start_idx = build_trial_prompt(tokenizer)
seq_len = input_ids.shape[1]
print(f"Prompt: {seq_len} tokens, injection starts at token {injection_start_idx}")

LAYER = 40
CONCEPT = "lightning"
ALPHA = 4

vec = load_concept_vector(VECTOR_DIR, CONCEPT, LAYER)
print(f"Vector loaded: {CONCEPT} layer {LAYER}, shape {vec.shape}, dtype {vec.dtype}")

# --- Control trial (alpha=0) ---
print("\n--- Control trial ---")
with model.generate(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=1.0,
    remote=True,
) as tracer:
    output_ctrl = tracer.result.save()

ctrl_response = tokenizer.decode(
    output_ctrl[0][seq_len:], skip_special_tokens=True
).strip()
print(f"Response:\n{ctrl_response}")

# --- Injection trial ---
print(f"\n--- Injection trial (alpha={ALPHA}, concept={CONCEPT}) ---")
with model.generate(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=1.0,
    remote=True,
) as tracer:
    hs = model.model.layers[LAYER].output[0]
    scaled = ALPHA * vec.to(device=hs.device, dtype=hs.dtype)
    injection = torch.zeros_like(hs)
    injection[injection_start_idx:, :] = scaled
    model.model.layers[LAYER].output[0] = hs + injection

    output_inj = tracer.result.save()

inj_response = tokenizer.decode(
    output_inj[0][seq_len:], skip_special_tokens=True
).strip()
print(f"Response:\n{inj_response}")

print("\n--- Done ---")
