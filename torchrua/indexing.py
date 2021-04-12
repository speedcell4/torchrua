from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.utils import get_device, accumulate_batch_sizes, resize_batch_sizes
from torchrua.utils import packed_sequence_to_lengths

__all__ = [
    'batch_sizes_to_ptr', 'lengths_to_ptr',
    'head_indices', 'select_head',
    'last_indices', 'select_last',
    'init_indices', 'select_init',
    'tail_indices', 'select_tail',
    'reversed_indices', 'reverse_packed_sequence',
    'rolled_indices', 'roll_packed_sequence',
]


@torch.no_grad()
def batch_sizes_to_ptr(
        batch_sizes: Tensor,
        sorted_indices: Optional[Tensor], unsorted_indices: Optional[Tensor],
        total_length: Optional[int], device: torch.device):
    batch_size = batch_sizes[0].item()
    if total_length is None:
        total_length = batch_sizes.size(0)

    batch_ptr = torch.arange(batch_size, device=device)
    token_ptr = torch.arange(total_length, device=device)

    tb_mask = batch_ptr[None, :] < resize_batch_sizes(batch_sizes, total_length)[:, None]

    if sorted_indices is not None:
        batch_ptr = sorted_indices

    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)

    lengths = tb_mask.long().sum(dim=0)
    if unsorted_indices is not None:
        lengths = lengths[unsorted_indices]

    return batch_ptr, token_ptr, lengths


@torch.no_grad()
def lengths_to_ptr(lengths: Tensor, sorted_indices: Optional[Tensor], device: torch.device):
    batch_size = lengths.size(0)
    num_tokens = lengths.max().item()

    batch_ptr = torch.arange(batch_size, device=device)
    token_ptr = torch.arange(num_tokens, device=device)

    tb_mask = token_ptr[:, None] < lengths[None, :]

    if sorted_indices is not None:
        batch_ptr = sorted_indices

    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
    batch_sizes = tb_mask.long().sum(dim=1)

    return batch_ptr, token_ptr, batch_sizes


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True, *, device: torch.device = None) -> Tensor:
    device = get_device(pack, device=device)
    batch_size = pack.batch_sizes[0].item()

    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices
    return torch.arange(0, batch_size, device=device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def last_indices(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None, *,
                 device: torch.device = None) -> Tensor:
    device = get_device(pack, device=device)
    if lengths is None:
        lengths = packed_sequence_to_lengths(pack=pack, unsort=False)

    indices = accumulate_batch_sizes(pack.batch_sizes, device=device)[lengths - 1]
    unsorted_indices = head_indices(pack=pack, unsort=unsort, device=device)

    return indices[unsorted_indices] + unsorted_indices


def select_last(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None) -> Tensor:
    return pack.data[last_indices(pack=pack, unsort=unsort, lengths=lengths)]


@torch.no_grad()
def init_indices(pack: PackedSequence, drop_last_n: int = 1, *, device: torch.device = None) -> Tensor:
    device = get_device(pack, device=device)
    total_length = pack.batch_sizes.size(0) - drop_last_n

    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(
        batch_sizes=pack.batch_sizes.to(device=device),
        sorted_indices=None,
        unsorted_indices=None,
        total_length=total_length, device=device,
    )

    indices = accumulate_batch_sizes(pack.batch_sizes, device=device)
    return indices[token_ptr] + batch_ptr


def select_init(pack: PackedSequence, drop_last_n: int = 1) -> PackedSequence:
    return PackedSequence(
        data=pack.data[init_indices(pack, drop_last_n=drop_last_n)],
        batch_sizes=pack.batch_sizes[drop_last_n:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(pack: PackedSequence, drop_first_n: int = 1, *,
                 device: torch.device = None) -> Tensor:
    assert pack.batch_sizes[0] == pack.batch_sizes[drop_first_n], \
        'some sequences contain only one element, truncating is not allowed.'

    device = get_device(pack, device=device)
    return torch.arange(
        pack.batch_sizes[0].item() * drop_first_n,
        pack.batch_sizes.sum().item(),
        device=device,
    )


def select_tail(pack: PackedSequence, drop_first_n: int = 1) -> PackedSequence:
    assert pack.batch_sizes[0] == pack.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    return PackedSequence(
        data=pack.data[pack.batch_sizes[0].item() * drop_first_n:],
        batch_sizes=pack.batch_sizes[drop_first_n:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def reversed_indices(pack: PackedSequence, *, device: torch.device = None) -> Tensor:
    device = get_device(pack, device=device)

    batch_ptr, token_ptr, lengths = batch_sizes_to_ptr(
        batch_sizes=pack.batch_sizes.to(device=device),
        sorted_indices=None,
        unsorted_indices=None,
        total_length=None, device=device,
    )
    token_ptr = (lengths - 1)[batch_ptr] - token_ptr

    indices = accumulate_batch_sizes(pack.batch_sizes, device=device)
    return indices[token_ptr] + batch_ptr


def reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=pack.data[reversed_indices(pack)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def rolled_indices(pack: PackedSequence, offset: int, *, device: torch.device = None) -> Tensor:
    device = get_device(pack, device=device)

    batch_ptr, token_ptr, lengths = batch_sizes_to_ptr(
        batch_sizes=pack.batch_sizes.to(device=device),
        sorted_indices=None,
        unsorted_indices=None,
        total_length=None, device=device,
    )
    lengths = lengths[batch_ptr]
    token_ptr = (token_ptr - offset + lengths) % lengths

    indices = accumulate_batch_sizes(pack.batch_sizes, device=device)
    return indices[token_ptr] + batch_ptr


def roll_packed_sequence(pack: PackedSequence, offset: int) -> PackedSequence:
    return PackedSequence(
        data=pack.data[rolled_indices(pack, offset=offset)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
