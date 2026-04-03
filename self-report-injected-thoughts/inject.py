"""
Contains the core experiment logic from the first experiment in Lindsey's protocol.


"""

import torch


def inject_at_layer(
    model: torch.nn.Module,
    layer_idx: int,
    alpha: int,
    input_ids: torch.Tensor,  # [1, seq_len]
    vector: torch.Tensor,  # [1, hidden]
) -> torch.nn.Module:
    pass
