from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import batch_sizes_to_ptr
from torchrua.utils import accumulate_sizes, resize_sizes, batch_sizes_to_token_sizes

__all__ = [
    'head_indices', 'select_head',
    'last_indices', 'select_last',
    'init_indices', 'select_init',
    'tail_indices', 'select_tail',
    'reverse_packed_indices', 'reverse_packed_sequence',
    'roll_packed_indices', 'roll_packed_sequence',
]


@torch.no_grad()
def head_indices(batch_sizes: Tensor, unsorted_indices: Optional[Tensor] = None) -> Tensor:
    if unsorted_indices is not None:
        return unsorted_indices

    return torch.arange(batch_sizes[0].item(), device=batch_sizes.device)


def select_head(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    device = sequence.data.device

    indices = head_indices(
        batch_sizes=sequence.batch_sizes.to(device=device),
        unsorted_indices=sequence.unsorted_indices if unsort else None,
    )
    return sequence.data[indices]


@torch.no_grad()
def last_indices(batch_sizes: Tensor, unsorted_indices: Optional[Tensor] = None) -> Tensor:
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr = head_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices)
    token_ptr = batch_sizes_to_token_sizes(batch_sizes=batch_sizes, batch_ptr=batch_ptr) - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_last(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    device = sequence.data.device

    indices = last_indices(
        batch_sizes=sequence.batch_sizes.to(device=device),
        unsorted_indices=sequence.unsorted_indices if unsort else None,
    )
    return sequence.data[indices]


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
def reverse_packed_indices(sequence: PackedSequence) -> Tensor:
    device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    token_ptr, batch_ptr, token_sizes = batch_sizes_to_ptr(batch_sizes=batch_sizes)
    token_ptr = token_sizes[batch_ptr] - token_ptr - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def reverse_packed_sequence(sequence: PackedSequence) -> PackedSequence:
    indices = reverse_packed_indices(sequence)
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


@torch.no_grad()
def roll_packed_indices(sequence: PackedSequence, shifts: int) -> Tensor:
    device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    token_ptr, batch_ptr, token_sizes = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    token_sizes = token_sizes[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return acc_batch_sizes[token_ptr] + batch_ptr


def roll_packed_sequence(sequence: PackedSequence, shifts: int) -> PackedSequence:
    indices = roll_packed_indices(sequence, shifts=shifts)
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
