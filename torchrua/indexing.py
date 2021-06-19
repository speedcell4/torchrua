from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.utils import accumulate_sizes, resize_sizes, batch_sizes_to_token_sizes

__all__ = [
    'batch_sizes_to_ptr',
    'token_sizes_to_ptr',
    'head_indices', 'select_head',
    'last_indices', 'select_last',
    'init_indices', 'select_init',
    'tail_indices', 'select_tail',
    'reversed_indices', 'reverse_packed_sequence',
    'rolled_indices', 'roll_packed_sequence',
]


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


@torch.no_grad()
def head_indices(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    device = sequence.data.device

    if unsort and sequence.unsorted_indices is not None:
        return sequence.unsorted_indices.to(device=device)

    b = sequence.batch_sizes[0].item()
    return torch.arange(0, b, device=device)


def select_head(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    return sequence.data[head_indices(sequence=sequence, unsort=unsort)]


@torch.no_grad()
def last_indices(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    batch_ptr = head_indices(sequence=sequence, unsort=unsort)
    token_ptr = batch_sizes_to_token_sizes(batch_sizes=batch_sizes, batch_ptr=batch_ptr) - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_last(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    return sequence.data[last_indices(sequence=sequence, unsort=unsort)]


@torch.no_grad()
def init_indices(sequence: PackedSequence, drop_last_n: int = 1) -> Tensor:
    device = sequence.data.device
    n = sequence.batch_sizes.size()[0] - drop_last_n

    batch_sizes = sequence.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    batch_sizes = resize_sizes(sizes=batch_sizes, n=n)
    token_ptr, batch_ptr, _ = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_init(sequence: PackedSequence, drop_last_n: int = 1) -> PackedSequence:
    return PackedSequence(
        data=sequence.data[init_indices(sequence, drop_last_n=drop_last_n)],
        batch_sizes=sequence.batch_sizes[drop_last_n:],
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(sequence: PackedSequence, drop_first_n: int = 1) -> Tensor:
    assert sequence.batch_sizes[0] == sequence.batch_sizes[drop_first_n], \
        f'some sequences contain less than {drop_first_n} elements, truncating is not allowed.'

    device = sequence.data.device
    return torch.arange(
        sequence.batch_sizes[0].item() * drop_first_n,
        sequence.batch_sizes.sum().item(),
        device=device,
    )


def select_tail(sequence: PackedSequence, drop_first_n: int = 1) -> PackedSequence:
    assert sequence.batch_sizes[0] == sequence.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    return PackedSequence(
        data=sequence.data[tail_indices(sequence, drop_first_n=drop_first_n)],
        batch_sizes=sequence.batch_sizes[drop_first_n:],
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


@torch.no_grad()
def reversed_indices(sequence: PackedSequence) -> Tensor:
    device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    token_ptr, batch_ptr, sorted_lengths = batch_sizes_to_ptr(batch_sizes=batch_sizes)
    token_ptr = (sorted_lengths - 1)[batch_ptr] - token_ptr

    return acc_batch_sizes[token_ptr] + batch_ptr


def reverse_packed_sequence(sequence: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=sequence.data[reversed_indices(sequence)],
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


@torch.no_grad()
def rolled_indices(sequence: PackedSequence, shifts: int) -> Tensor:
    device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    token_ptr, batch_ptr, sorted_lengths = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    lengths = sorted_lengths[batch_ptr]
    token_ptr = (token_ptr - shifts + lengths) % lengths

    return acc_batch_sizes[token_ptr] + batch_ptr


def roll_packed_sequence(sequence: PackedSequence, shifts: int) -> PackedSequence:
    return PackedSequence(
        data=sequence.data[rolled_indices(sequence, shifts=shifts)],
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
