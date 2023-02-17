from functools import singledispatch
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import accumulate_sizes, major_sizes_to_ptr, CattedSequence

__all__ = [
    'trunc_sequence',
    'trunc_catted_indices', 'trunc_catted_sequence',
    'trunc_packed_indices', 'trunc_packed_sequence',
]


@singledispatch
def trunc_sequence(sequence: Union[CattedSequence, PackedSequence], trunc: Tuple[int, int]):
    raise TypeError(f'type {type(sequence)} is not supported.')


@torch.no_grad()
def trunc_catted_indices(token_sizes: Tensor, trunc: Tuple[int, int], device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes = token_sizes - trunc[0] - trunc[1]
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    return acc_token_sizes[batch_ptr] + token_ptr + trunc[0], token_sizes


@trunc_sequence.register
def trunc_catted_sequence(sequence: CattedSequence, trunc: Tuple[int, int]) -> CattedSequence:
    indices, token_sizes = trunc_catted_indices(
        token_sizes=sequence.token_sizes, trunc=trunc,
        device=sequence.data.device,
    )

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=token_sizes,
    )


@torch.no_grad()
def trunc_packed_indices(batch_sizes: Tensor, trunc: Tuple[int, int], device: Device = None):
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_sizes = batch_sizes[trunc[0] + trunc[1]:]
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)

    return batch_ptr + acc_batch_sizes[token_ptr + trunc[0]], batch_sizes


@trunc_sequence.register
def trunc_packed_sequence(sequence: PackedSequence, trunc: Tuple[int, int]) -> PackedSequence:
    indices, batch_sizes = trunc_packed_indices(
        batch_sizes=sequence.batch_sizes, trunc=trunc,
        device=sequence.data.device,
    )

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
