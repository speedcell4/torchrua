from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import major_sizes_to_ptr, accumulate_sizes, CattedSequence, major_masked_select

__all__ = [
    'roll_sequence', 'roll_indices',
    'roll_catted_indices',
    'roll_packed_indices',
]


def roll_indices(sequence: Union[CattedSequence, PackedSequence], shifts: int):
    if isinstance(sequence, CattedSequence):
        return roll_catted_indices(
            token_sizes=sequence.token_sizes,
            shifts=shifts, device=sequence.data.device,
        )

    if isinstance(sequence, PackedSequence):
        return roll_packed_indices(
            batch_sizes=sequence.batch_sizes,
            shifts=shifts, device=sequence.data.device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def roll_catted_indices(token_sizes: Tensor, shifts: int, device: Device = None) -> Tensor:
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    token_sizes = torch.repeat_interleave(token_sizes, repeats=token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return acc_token_sizes[batch_ptr] + token_ptr


def roll_packed_indices(batch_sizes: Tensor, shifts: int, device: Device = None) -> Tensor:
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr, token_sizes = major_masked_select(sizes=batch_sizes, device=device)
    token_sizes = token_sizes[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return batch_ptr + acc_batch_sizes[token_ptr]


def roll_sequence(sequence: Union[CattedSequence, PackedSequence], shifts: int):
    indices = roll_indices(sequence=sequence, shifts=shifts)

    return sequence._replace(data=sequence.data[indices])
