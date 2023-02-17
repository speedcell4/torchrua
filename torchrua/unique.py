from functools import singledispatch
from typing import Union

import torch
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import major_sizes_to_ptr, CattedSequence

__all__ = [
    'unique_sequence',
    'unique_catted_sequence',
    'unique_packed_sequence',
]


@singledispatch
def unique_sequence(sequence: Union[CattedSequence, PackedSequence], device: Device = None):
    raise TypeError(f'type {type(sequence)} is not supported')


@unique_sequence.register
def unique_catted_sequence(sequence: CattedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    data = sequence.data.to(device=device)
    token_sizes = sequence.token_sizes.to(device=device)

    unique1, token_ptr = torch.unique(data, sorted=True, return_inverse=True)
    batch_ptr = torch.repeat_interleave(token_sizes)

    n = unique1.size()[0]
    unique2, inverse_ptr, counts = torch.unique(
        n * batch_ptr + token_ptr,
        sorted=True, return_inverse=True, return_counts=True,
    )

    return unique1[unique2 % n], inverse_ptr, counts


@unique_sequence.register
def unique_packed_sequence(sequence: PackedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    data = sequence.data.to(device=device)
    batch_sizes = sequence.batch_sizes.to(device=device)

    unique1, token_ptr = torch.unique(data, sorted=True, return_inverse=True)
    batch_ptr, _ = major_sizes_to_ptr(sizes=batch_sizes)

    n = unique1.size()[0]
    unique2, inverse_ptr, counts = torch.unique(
        n * batch_ptr + token_ptr,
        sorted=True, return_inverse=True, return_counts=True,
    )

    return unique1[unique2 % n], inverse_ptr, counts
