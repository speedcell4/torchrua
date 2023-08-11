from functools import singledispatch
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import CattedSequence
from torchrua.core import accumulate_sizes
from torchrua.core import major_masked_select
from torchrua.core import major_sizes_to_ptr

__all__ = [
    'reverse_sequence',
    'reverse_catted_indices', 'reverse_catted_sequence',
    'reverse_packed_indices', 'reverse_packed_sequence',
]


@singledispatch
def reverse_sequence(sequence: Union[CattedSequence, PackedSequence]):
    raise TypeError(f'type {type(sequence)} is not supported')


@torch.no_grad()
def reverse_catted_indices(token_sizes: Tensor, device: Device = None) -> Tensor:
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    token_ptr = (token_sizes - 1)[batch_ptr] - token_ptr

    return acc_token_sizes[batch_ptr] + token_ptr


@reverse_sequence.register
def reverse_catted_sequence(sequence: CattedSequence) -> CattedSequence:
    indices = reverse_catted_indices(token_sizes=sequence.token_sizes, device=sequence.data.device)

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )


@torch.no_grad()
def reverse_packed_indices(batch_sizes: Tensor, device: Device = None) -> Tensor:
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr, token_sizes = major_masked_select(sizes=batch_sizes)
    token_ptr = (token_sizes - 1)[batch_ptr] - token_ptr

    return batch_ptr + acc_batch_sizes[token_ptr]


@reverse_sequence.register
def reverse_packed_sequence(sequence: PackedSequence) -> PackedSequence:
    indices = reverse_packed_indices(batch_sizes=sequence.batch_sizes, device=sequence.data.device)

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
