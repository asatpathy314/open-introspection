# Llama-3.1-8B Vectors Storage

This repo keeps only the precomputed Llama-3.1-8B concept vectors.

## Location
- Vectors root: `data/vectors/llama-8b`

## File naming
Each vector is stored as a single `.pt` file named:
- `L{layer}_{concept}_{position}.pt`

Examples:
- `L12_ambition_last.pt`
- `L12_cleopatra_last.pt`

## Storage format
Vectors are saved with PyTorch:
- `torch.save(vector.detach().cpu().float(), path)`
- This means the tensor is stored on CPU as `float32`.

## Notes
- The cache path convention is implemented in `src/introspection/vectors/cache.py` in the original codebase.
- `concept` is lowercased and sanitized (spaces and `/` become `_`).
- `position` is typically `last`.
