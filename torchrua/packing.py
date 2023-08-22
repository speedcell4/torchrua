from typing import Tuple
from typing import Union

import torch
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_sequence
from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import get_device
from torchrua.core import sizes_to_sorting
from torchrua.info import token_sizes_to_minor_ptr3
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T
from torchrua.ty import Ts
from torchrua.ty import is_type

__all__ = [
    'pack_sequence', 'pack_indices',
    'pack_catted_indices', 'pack_padded_indices',
]


def pack_sequence(sequence: Union[Ts, C, D], device: torch.device = None) -> P:
    if is_type(sequence, Ts):
        sequence = cat_sequence(sequence=sequence, device=device)

    device = get_device(*sequence, device=device)
    indices, batch_sizes, sorted_indices, unsorted_indices = pack_indices(sequence, device=device)

    return PackedSequence(
        data=sequence[0][indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


C.pack = pack_sequence
P.pack = P.to


def pack_indices(sequence: Union[C, D], device: torch.device = None):
    if is_type(sequence, C):
        return pack_catted_indices(
            token_sizes=sequence.token_sizes,
            device=device,
        )

    if is_type(sequence, D):
        return pack_padded_indices(
            token_sizes=sequence[1],
            device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def pack_catted_indices(token_sizes: T, device: torch.device = None) -> Tuple[T, T, T, T]:
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(sizes=token_sizes, device=device)
    _, (batch_ptr, token_ptr), (batch_sizes, _) = token_sizes_to_minor_ptr3(
        token_sizes=token_sizes, batch_ptr=sorted_indices,
    )

    return acc_token_sizes[batch_ptr] + token_ptr, batch_sizes, sorted_indices, unsorted_indices


def pack_padded_indices(token_sizes: T, device: torch.device = None) -> Tuple[Tuple[T, T], T, T, T]:
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(sizes=token_sizes, device=device)
    _, (batch_ptr, token_ptr), (batch_sizes, _) = token_sizes_to_minor_ptr3(
        token_sizes=sorted_token_sizes, batch_ptr=sorted_indices,
    )

    return (batch_ptr, token_ptr), batch_sizes, sorted_indices, unsorted_indices
