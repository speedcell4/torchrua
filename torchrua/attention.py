from typing import Tuple

import torch
from torch import Tensor

from torchrua import batch_sizes_to_ptr, accumulate_sizes
from torchrua.scatter import scatter_index_to_ptr

__all__ = [
    'attention_indices',
    'scatter_multi_head_attention',
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


def scatter_multi_head_attention(query: Tensor, key: Tensor, value: Tensor, tau: float, q_ptr: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=q_ptr, device=query.device)
    score_view = (query[..., None, :] @ key[..., :, None] * tau).view((query.size()[0], -1))

    with torch.no_grad():
        m, _, _, _ = torch.embedding_bag(
            weight=score_view,
            indices=indices, offsets=offsets, mode=2,
        )

    s, _, _, _ = torch.embedding_bag(
        weight=(score_view - m[q_ptr]).exp(),
        indices=indices, offsets=offsets, mode=0,
    )
    log_partition = torch.masked_fill(s, s == 0, 1.).log() + m

    attention = (score_view - log_partition[q_ptr]).exp()
    ret, _, _, _ = torch.embedding_bag(
        weight=(attention[..., None] * value).view((attention.size()[0], -1)),
        indices=indices, offsets=offsets, mode=0,
    )

    return ret.view((ret.size()[0], *value.size()[1:]))
