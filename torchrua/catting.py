from typing import Tuple
from typing import Union

import torch

from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import get_device
from torchrua.info import batch_sizes_to_minor_ptr3
from torchrua.info import token_sizes_to_major_ptr3
from torchrua.ty import C
from torchrua.ty import CattedSequence
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T
from torchrua.ty import Ts
from torchrua.ty import is_type

__all__ = [
    'cat_sequence', 'cat_indices',
    'cat_packed_indices', 'cat_padded_indices',
]


def cat_sequence(sequence: Union[Ts, D, P], device: torch.device = None) -> C:
    if is_type(sequence, Ts):
        device = get_device(*sequence, device=device)

        return CattedSequence(
            data=torch.cat(sequence, dim=0).to(device=device),
            token_sizes=torch.tensor([s.size()[0] for s in sequence], dtype=torch.long, device=device),
        )

    indices, token_sizes = cat_indices(sequence=sequence, device=device)
    return CattedSequence(data=sequence[0][indices], token_sizes=token_sizes)


C.cat = C.to
P.cat = cat_sequence


def cat_indices(sequence: Union[D, P], device: torch.device = None):
    if is_type(sequence, D):
        return cat_padded_indices(
            token_sizes=sequence[1],
            device=device,
        )

    if is_type(sequence, P):
        return cat_packed_indices(
            batch_sizes=sequence.batch_sizes,
            unsorted_indices=sequence.unsorted_indices,
            device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def cat_padded_indices(token_sizes: T, device: torch.device = None) -> Tuple[Tuple[T, T], T]:
    _, (batch_ptr, token_ptr), (_, token_sizes) = token_sizes_to_major_ptr3(token_sizes, device=device)
    return (batch_ptr, token_ptr), token_sizes


def cat_packed_indices(batch_sizes: T, unsorted_indices: T, device: torch.device = None) -> Tuple[T, T]:
    unsorted_indices, batch_sizes, device = broadcast_devices(unsorted_indices, batch_sizes, device=device)

    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    _, (batch_ptr, token_ptr), (_, token_sizes) = batch_sizes_to_minor_ptr3(
        batch_sizes=batch_sizes, batch_ptr=unsorted_indices,
    )

    return batch_ptr + acc_batch_sizes[token_ptr], token_sizes
