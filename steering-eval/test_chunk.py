"""Debug: test different ways to save multiple layer outputs."""
import nnsight, os, torch
from dotenv import load_dotenv
load_dotenv()
nnsight.CONFIG.set_default_api_key(os.environ['NDIF_API_KEY'])
model = nnsight.LanguageModel('meta-llama/Llama-3.1-70B-Instruct')

messages = [{"role": "user", "content": "Tell me about dust."}]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Test: separate variable names (known to work)
with model.trace(prompt, remote=True):
    h0 = model.model.layers[0].output[0][-1, :].save()
    h1 = model.model.layers[1].output[0][-1, :].save()
    h2 = model.model.layers[2].output[0][-1, :].save()

print(f"h0: shape={h0.shape}")
print(f"h1: shape={h1.shape}")
print(f"h2: shape={h2.shape}")

# Test: pre-allocated list
saves = [None] * 5
with model.trace(prompt, remote=True):
    saves[0] = model.model.layers[0].output[0][-1, :].save()
    saves[1] = model.model.layers[1].output[0][-1, :].save()
    saves[2] = model.model.layers[2].output[0][-1, :].save()
    saves[3] = model.model.layers[3].output[0][-1, :].save()
    saves[4] = model.model.layers[4].output[0][-1, :].save()

for i in range(5):
    v = saves[i]
    if hasattr(v, 'shape'):
        print(f"List[{i}]: shape={v.shape}")
    else:
        print(f"List[{i}]: type={type(v)}, val={v}")
