import os
from dotenv import load_dotenv

from nnsight import LanguageModel, CONFIG

# Retrieve the NDIF_API_KEY and HF_API_KEY
load_dotenv()

api_key = os.getenv("NDIF_API_KEY")

if not api_key:
    raise RuntimeError("Set NDIF_API_KEY in your environment before running this script.")

CONFIG.set_default_api_key(api_key)
# Load model: We'll never actually load the parameters so no need to specify a device_map.
model = LanguageModel("openai-community/gpt2")

# To specify using NDIF remotely instead of executing locally, set remote=True.
with model.trace("The Eiffel Tower is in the city of", remote=True):
    hidden_state = model.transformer.h[3].output.save()
    output = model.output.save()
