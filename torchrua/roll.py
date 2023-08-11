from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import CattedSequence
from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import major_sizes_to_ptr
from torchrua.info import token_sizes_to_major_ptr3

__all__ = [
    'roll_indices', 'roll_sequence',
    'roll_catted_indices',
    'roll_packed_indices',
]

Sequence = Union[CattedSequence, PackedSequence]


def roll_indices(sequence: Sequence, shifts: int):
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


def roll_catted_indices(token_sizes: Tensor, shifts: int, device: torch.device = None) -> Tensor:
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    token_sizes = torch.repeat_interleave(token_sizes, repeats=token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return acc_token_sizes[batch_ptr] + token_ptr


def roll_packed_indices(batch_sizes: Tensor, shifts: int, device: torch.device = None) -> Tensor:
    batch_sizes, device = broadcast_devices(batch_sizes, device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    _, (token_ptr, batch_ptr), (token_sizes, _) = token_sizes_to_major_ptr3(batch_sizes, device=device)
    token_sizes = token_sizes[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return batch_ptr + acc_batch_sizes[token_ptr]


def roll_sequence(sequence: Sequence, shifts: int):
    indices = roll_indices(sequence=sequence, shifts=shifts)

    return sequence._replace(data=sequence.data[indices])
