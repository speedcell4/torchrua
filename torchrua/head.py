from typing import Union

import torch

from torchrua import broadcast_devices
from torchrua.core import accumulate_sizes
from torchrua.ty import C
from torchrua.ty import P
from torchrua.ty import T
from torchrua.ty import is_type

__all__ = [
    'head_sequence', 'head_indices',
    'head_catted_indices', 'head_packed_indices',
]


def head_sequence(sequence: Union[C, P]) -> T:
    return sequence.data[head_indices(sequence=sequence)]


C.head = head_sequence
P.head = head_sequence


def head_indices(sequence: Union[C, P]) -> T:
    if is_type(sequence, C):
        return head_catted_indices(
            token_sizes=sequence.token_sizes,
            device=sequence.data.device,
        )

    if is_type(sequence, P):
        return head_packed_indices(
            unsorted_indices=sequence.unsorted_indices,
            device=sequence.data.device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def head_catted_indices(token_sizes: T, device: torch.device = None) -> T:
    token_sizes, _ = broadcast_devices(token_sizes, device=device)
    return accumulate_sizes(sizes=token_sizes)


def head_packed_indices(unsorted_indices: T, device: torch.device = None) -> T:
    unsorted_indices, _ = broadcast_devices(unsorted_indices, device=device)
    return unsorted_indices
