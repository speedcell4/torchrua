from typing import Tuple, Optional

import torch
from torch import Tensor

from torchrua.utils import accumulate_sizes


@torch.no_grad()
def major_sizes_to_ptr(sizes: Tensor) -> Tuple[Tensor, Tensor]:
    minor_ptr = torch.repeat_interleave(repeats=sizes)

    major_ptr = torch.repeat_interleave(accumulate_sizes(sizes), repeats=sizes)
    major_ptr = torch.arange(major_ptr.size()[0], device=major_ptr.device) - major_ptr

    return major_ptr, minor_ptr


@torch.no_grad()
def transpose_sizes(sizes: Tensor) -> Tensor:
    index = torch.arange(sizes.max().item(), device=sizes.device)
    return (index[:, None] < sizes[None, :]).long().sum(dim=-1)


@torch.no_grad()
def batch_sizes_to_ptr(batch_sizes: Tensor,
                       token_ptr: Optional[Tensor] = None,
                       batch_ptr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    t = batch_sizes.size()[0]
    b = batch_sizes.max().item()

    if token_ptr is None:
        token_ptr = torch.arange(t, device=batch_sizes.device)
    assert token_ptr.size() == (t,)

    if batch_ptr is None:
        batch_ptr = torch.arange(b, device=batch_sizes.device)
    assert batch_ptr.size() == (b,)

    tb_mask = batch_ptr[None, :] < batch_sizes[:, None]

    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    sorted_token_sizes = tb_mask.long().sum(dim=0)

    return token_ptr, batch_ptr, sorted_token_sizes


@torch.no_grad()
def token_sizes_to_ptr(token_sizes: Tensor,
                       token_ptr: Optional[Tensor] = None,
                       batch_ptr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    t = token_sizes.max().item()
    b = token_sizes.size()[0]

    if token_ptr is None:
        token_ptr = torch.arange(t, device=token_sizes.device)
    assert token_ptr.size() == (t,)

    if batch_ptr is None:
        batch_ptr = torch.arange(b, device=token_sizes.device)
    assert batch_ptr.size() == (b,)

    tb_mask = token_ptr[:, None] < token_sizes[None, :]

    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    sorted_batch_sizes = tb_mask.long().sum(dim=1)

    return token_ptr, batch_ptr, sorted_batch_sizes
