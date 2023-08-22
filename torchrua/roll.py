from typing import Union

import torch

from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import major_sizes_to_ptr
from torchrua.info import token_sizes_to_major_ptr3
from torchrua.ty import C
from torchrua.ty import P
from torchrua.ty import T
from torchrua.ty import is_type

__all__ = [
    'roll_sequence', 'roll_indices',
    'roll_catted_indices', 'roll_packed_indices',
]


def roll_sequence(sequence: Union[C, P], shifts: int) -> Union[C, P]:
    indices = roll_indices(sequence=sequence, shifts=shifts, device=sequence.data.device)
    return sequence._replace(data=sequence.data[indices])


C.roll = roll_sequence
P.roll = roll_sequence


def roll_indices(sequence: Union[C, P], shifts: int, device: torch.device = None) -> T:
    if is_type(sequence, C):
        return roll_catted_indices(
            token_sizes=sequence.token_sizes,
            shifts=shifts, device=device,
        )

    if is_type(sequence, P):
        return roll_packed_indices(
            batch_sizes=sequence.batch_sizes,
            shifts=shifts, device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def roll_catted_indices(token_sizes: T, shifts: int, device: torch.device = None) -> T:
    token_sizes, device = broadcast_devices(token_sizes, device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    token_sizes = torch.repeat_interleave(token_sizes, repeats=token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return acc_token_sizes[batch_ptr] + token_ptr


def roll_packed_indices(batch_sizes: T, shifts: int, device: torch.device = None) -> T:
    batch_sizes, device = broadcast_devices(batch_sizes, device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    _, (token_ptr, batch_ptr), (token_sizes, _) = token_sizes_to_major_ptr3(batch_sizes, device=device)
    token_sizes = token_sizes[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return batch_ptr + acc_batch_sizes[token_ptr]
