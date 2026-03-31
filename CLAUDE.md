# CLAUDE.md

## nnsight: Do NOT Iterate Over Envoy-Wrapped ModuleLists

Iterating over an Envoy (e.g., `for layer in model.model.layers`) or accessing `._children` causes a `RecursionError` due to the `__iter__` → `_children` → `__dict__.values()` → `__getattr__` recursion chain. This happens both inside and outside trace contexts. List comprehensions with index-based access also fail.

```python
# WRONG - all of these cause RecursionError
[layer for layer in model.model.layers]           # direct iteration
len(model.model.layers._children)                 # accessing _children
[model.model.layers[i] for i in range(32)]        # list comprehension with indexing

# CORRECT - use a for loop with explicit append and .save() the list
with model.trace(prompt, remote=True) as tracer:
    hidden_states = []
    for i in range(num_layers):  # num_layers must be known ahead of time
        hidden_states.append(model.model.layers[i].output[0].detach().cpu().save())
    hidden_states.save()  # must save the list itself to access it after the trace

# CORRECT - don't pass modules to cache (caches everything, but downloads a lot of data)
with model.trace(prompt) as tracer:
    cache = tracer.cache().save()
```
