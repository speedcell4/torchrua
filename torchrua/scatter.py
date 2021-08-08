from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import invert_permutation
from torch.types import Device

__all__ = [
    'scatter_index_to_ptr',
]


@torch.no_grad()
def scatter_index_to_ptr(index: Tensor, keep_order: bool = True,
                         dtype: torch.dtype = torch.long,
                         device: Device = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if device is None:
        device = index.device

    sorted_index, sorted_indices = torch.sort(index, dim=0, stable=True, descending=False)
    unsorted_indices = invert_permutation(sorted_indices)

    other_ptr = torch.arange(index.max().item() + 1, dtype=dtype, device=device)
    tb_mask = other_ptr[:, None] == index[None, :]
    acc_mask = tb_mask.to(dtype=dtype).cumsum(dim=1)

    token_sizes = acc_mask[:, -1]
    other_ptr = torch.masked_select(acc_mask, mask=tb_mask) - 1

    if not keep_order:
        return sorted_index, other_ptr, sorted_indices, unsorted_indices, token_sizes
    else:
        return index, other_ptr[unsorted_indices], sorted_indices, unsorted_indices, token_sizes
