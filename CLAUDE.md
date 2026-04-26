# CLAUDE.md

## nnsight 0.6.2 — Envoy iteration and trace-body semantics

A few traps in `nnsight==0.6.2` (the pinned version). Verified empirically against NDIF on Llama-3.1-70B-Instruct on 2026-04-26.

### 1. Don't iterate Envoy-wrapped ModuleLists

Direct iteration or list comprehensions over an Envoy (e.g. `model.model.layers`) trigger a `RecursionError` in `tracer.py:__getattr__`. This fires both inside and outside trace contexts. Use index-based access in a `range()` loop with `num_layers` known ahead of time.

```python
# WRONG — RecursionError
[layer for layer in model.model.layers]
len(model.model.layers._children)

# CORRECT — index into the Envoy
[model.model.layers[i] for i in range(num_layers)]
```

`tracer.cache(modules=...)` and even `tracer.cache()` with no args also hit the same recursion. Don't use `tracer.cache()` in this version — save tensors explicitly instead.

### 2. The body of `with model.trace(...)` is captured, not executed

nnsight rewrites the `with` block via source analysis. Variables assigned inside aren't bound in the outer scope, and `list.append(...)` calls do not mutate outer lists. Only single-name `.save()` assignments survive.

```python
# WRONG — `hidden_states` not bound after the trace; `.append()` is dropped
hidden_states = []
with model.trace(prompt, remote=True):
    for i in range(num_layers):
        hidden_states.append(model.model.layers[i].output[0][-1, :].save())
# UnboundLocalError or empty list

# CORRECT — stack inside the trace and save a single tensor
with model.trace(prompt, remote=True):
    stacked = torch.stack(
        [model.model.layers[i].output[0][-1, :].cpu() for i in range(num_layers)],
        dim=0,
    ).save()
result = stacked.detach()  # real tensor after trace exits
```

### 3. Layer output shape and device for sharded models

- `model.model.layers[i].output[0]` is `[seq_len, hidden_dim]` (2-D, no batch axis) with nnsight. Don't index `[0, -1, :]` — use `[-1, :]`.
- For multi-GPU sharded models on NDIF (e.g. 70B), different layers live on different CUDA devices. `torch.stack` across them errors with `Expected all tensors to be on the same device`. Add `.cpu()` per-layer before stacking.
