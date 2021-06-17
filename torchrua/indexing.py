from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.utils import accumulate_batch_sizes, resize_batch_sizes, packed_sequence_to_lengths

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
        sorted_indices: Optional[Tensor] = None,
        total_length: Optional[int] = None,
        device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = batch_sizes.device

    if total_length is None:
        total_length = batch_sizes.size()[0]
    batch_size = batch_sizes[0].item()

    batch_ptr = torch.arange(batch_size, device=device)
    token_ptr = torch.arange(total_length, device=device)

    tb_mask = batch_ptr[None, :] < resize_batch_sizes(batch_sizes, total_length)[:, None]

    if sorted_indices is not None:
        batch_ptr = sorted_indices

    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
    sorted_lengths = tb_mask.long().sum(dim=0)

    return batch_ptr, token_ptr, sorted_lengths


@torch.no_grad()
def lengths_to_ptr(lengths: Tensor,
                   sorted_indices: Optional[Tensor] = None,
                   device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = lengths.device

    batch_size = lengths.size()[0]
    total_length = lengths.max().item()

    batch_ptr = torch.arange(batch_size, device=device)
    token_ptr = torch.arange(total_length, device=device)

    tb_mask = token_ptr[:, None] < lengths[None, :]

    if sorted_indices is not None:
        batch_ptr = sorted_indices

    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
    batch_sizes = tb_mask.long().sum(dim=1)

    return batch_ptr, token_ptr, batch_sizes


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True, device: Optional[torch.device] = None) -> Tensor:
    if device is None:
        device = pack.data.device

    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices.to(device=device)

    batch_size = pack.batch_sizes[0].item()
    return torch.arange(0, batch_size, device=device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def last_indices(pack: PackedSequence, unsort: bool = True, device: Optional[torch.device] = None) -> Tensor:
    if device is None:
        device = pack.data.device

    acc_batch_sizes = accumulate_batch_sizes(pack.batch_sizes, device=device)
    batch_ptr = head_indices(pack=pack, unsort=unsort, device=device)
    token_ptr = packed_sequence_to_lengths(pack=pack, unsort=unsort) - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_last(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[last_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def init_indices(pack: PackedSequence, drop_last_n: int = 1, device: Optional[torch.device] = None) -> Tensor:
    if device is None:
        device = pack.data.device
    total_length = pack.batch_sizes.size()[0] - drop_last_n

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_batch_sizes(batch_sizes=batch_sizes)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(
        batch_sizes=batch_sizes,
        total_length=total_length,
    )

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_init(pack: PackedSequence, drop_last_n: int = 1) -> PackedSequence:
    return PackedSequence(
        data=pack.data[init_indices(pack, drop_last_n=drop_last_n)],
        batch_sizes=pack.batch_sizes[drop_last_n:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(pack: PackedSequence, drop_first_n: int = 1, device: Optional[torch.device] = None) -> Tensor:
    assert pack.batch_sizes[0] == pack.batch_sizes[drop_first_n], \
        f'some sequences contain less than {drop_first_n} elements, truncating is not allowed.'

    if device is None:
        device = pack.data.device

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
def reversed_indices(pack: PackedSequence, device: Optional[torch.device] = None) -> Tensor:
    if device is None:
        device = pack.data.device

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_batch_sizes(batch_sizes=batch_sizes)
    batch_ptr, token_ptr, sorted_lengths = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    token_ptr = (sorted_lengths - 1)[batch_ptr] - token_ptr

    return acc_batch_sizes[token_ptr] + batch_ptr


def reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=pack.data[reversed_indices(pack)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def rolled_indices(pack: PackedSequence, offset: int, device: Optional[torch.device] = None) -> Tensor:
    if device is None:
        device = pack.data.device

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_batch_sizes(batch_sizes=batch_sizes)
    batch_ptr, token_ptr, sorted_lengths = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    lengths = sorted_lengths[batch_ptr]
    token_ptr = (token_ptr - offset + lengths) % lengths

    return acc_batch_sizes[token_ptr] + batch_ptr


def roll_packed_sequence(pack: PackedSequence, offset: int) -> PackedSequence:
    return PackedSequence(
        data=pack.data[rolled_indices(pack, offset=offset)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
