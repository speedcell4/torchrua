from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import invert_permutation

__all__ = [
    'accumulate_sizes', 'resize_sizes', 'sizes_to_sorting_indices',
    'batch_sizes_to_mask', 'batch_sizes_to_token_sizes',
    'token_sizes_to_mask', 'token_sizes_to_batch_sizes',
]


@torch.no_grad()
def accumulate_sizes(sizes: Tensor) -> Tensor:
    return F.pad(sizes.cumsum(dim=0), pad=[1, -1])


@torch.no_grad()
def resize_sizes(sizes: Tensor, n: int) -> Tensor:
    if n <= sizes.size()[0]:
        assert sizes[0] == sizes[-n]
        return sizes[-n:]
    return F.pad(sizes, [n - sizes.size()[0], 0], value=sizes[0])


@torch.no_grad()
def sizes_to_sorting_indices(sizes: Tensor, descending: bool = True,
                             device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = sizes.device

    sorted_sizes, sorted_indices = sizes.cpu().sort(dim=0, descending=descending)
    sorted_sizes = sorted_sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)
    unsorted_indices = invert_permutation(sorted_indices)

    return sorted_sizes, sorted_indices, unsorted_indices


@torch.no_grad()
def batch_sizes_to_mask(batch_sizes: Tensor,
                        batch_ptr: Optional[Tensor] = None,
                        batch_first: bool = False,
                        dtype: torch.dtype = torch.bool) -> Tensor:
    b = batch_sizes.max().item()

    if batch_ptr is None:
        batch_ptr = torch.arange(b, device=batch_sizes.device)
    assert batch_ptr.size() == (b,)

    if batch_first:
        mask = batch_ptr[:, None] < batch_sizes[None, :]
    else:
        mask = batch_ptr[None, :] < batch_sizes[:, None]

    return mask.to(dtype=dtype)


@torch.no_grad()
def batch_sizes_to_token_sizes(batch_sizes: Tensor,
                               batch_ptr: Optional[Tensor] = None,
                               dtype: torch.dtype = torch.long) -> Tensor:
    return batch_sizes_to_mask(
        batch_sizes=batch_sizes,
        batch_ptr=batch_ptr,
        batch_first=False,
        dtype=dtype,
    ).sum(dim=0)


@torch.no_grad()
def token_sizes_to_mask(token_sizes: Tensor,
                        token_ptr: Optional[Tensor] = None,
                        token_first: bool = False,
                        dtype: torch.dtype = torch.bool) -> Tensor:
    t = token_sizes.max().item()

    if token_ptr is None:
        token_ptr = torch.arange(t, device=token_sizes.device)
    assert token_ptr.size() == (t,)

    if token_first:
        mask = token_ptr[None, :] < token_sizes[:, None]
    else:
        mask = token_ptr[:, None] < token_sizes[None, :]

    return mask.to(dtype=dtype)


@torch.no_grad()
def token_sizes_to_batch_sizes(token_sizes: Tensor,
                               token_ptr: Optional[Tensor] = None,
                               dtype: torch.dtype = torch.long) -> Tensor:
    return token_sizes_to_mask(
        token_sizes=token_sizes,
        token_ptr=token_ptr,
        token_first=False,
        dtype=dtype,
    ).sum(dim=1)
