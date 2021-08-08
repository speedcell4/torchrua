from typing import Tuple

import torch
from torch import Tensor

from torchrua import batch_sizes_to_ptr, accumulate_sizes

__all__ = [
    'attention_indices',
]


@torch.no_grad()
def attention_indices(q_token_sizes: Tensor, k_token_sizes: Tensor) -> Tuple[Tensor, Tensor]:
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=q_token_sizes * k_token_sizes)
    q_token_ptr = torch.div(token_ptr, k_token_sizes[batch_ptr], rounding_mode='trunc')
    k_token_ptr = token_ptr % k_token_sizes[batch_ptr]

    q_acc_token_sizes = accumulate_sizes(sizes=q_token_sizes)
    k_acc_token_sizes = accumulate_sizes(sizes=k_token_sizes)

    q_ptr = q_token_ptr + q_acc_token_sizes[batch_ptr]
    k_ptr = k_token_ptr + k_acc_token_sizes[batch_ptr]
    return q_ptr, k_ptr
