from functools import singledispatch

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_ptr, accumulate_sizes
from torchrua.wrapper import Sequence

__all__ = [
    'roll_sequence',
    'roll_catted_indices', 'roll_catted_sequence',
    'roll_packed_indices', 'roll_packed_sequence',
]


@singledispatch
def roll_sequence(sequence: Sequence, shifts: int) -> Sequence:
    raise KeyError(f'type {type(sequence)} is not supported')


@torch.no_grad()
def roll_catted_indices(token_sizes: Tensor, shifts: int, device: Device = None) -> Tensor:
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    token_sizes = token_sizes[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return token_ptr + acc_token_sizes[batch_ptr]


@roll_sequence.register
def roll_catted_sequence(sequence: CattedSequence, shifts: int) -> CattedSequence:
    indices = roll_catted_indices(sequence.token_sizes, shifts=shifts, device=sequence.data.device)

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )


@torch.no_grad()
def roll_packed_indices(batch_sizes: Tensor, shifts: int, device: Device = None) -> Tensor:
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    _, token_sizes = torch.unique(batch_ptr, sorted=True, return_counts=True)
    token_sizes = token_sizes[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return acc_batch_sizes[token_ptr] + batch_ptr


@roll_sequence.register
def roll_packed_sequence(sequence: PackedSequence, shifts: int) -> PackedSequence:
    indices = roll_packed_indices(sequence.batch_sizes, shifts=shifts, device=sequence.data.device)

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
