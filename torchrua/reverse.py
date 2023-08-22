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
    'reverse_sequence', 'reverse_indices',
    'reverse_catted_indices', 'reverse_packed_indices',
]


def reverse_sequence(sequence: Union[C, P]) -> Union[C, P]:
    indices = reverse_indices(sequence, device=sequence.data.device)
    return sequence._replace(data=sequence.data[indices])


C.rev = reverse_sequence
P.rev = reverse_sequence


def reverse_indices(sequence: Union[C, P], device: torch.device = None) -> T:
    if is_type(sequence, C):
        return reverse_catted_indices(
            token_sizes=sequence.token_sizes,
            device=device,
        )

    if is_type(sequence, P):
        return reverse_packed_indices(
            batch_sizes=sequence.batch_sizes,
            device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def reverse_catted_indices(token_sizes: T, device: torch.device = None) -> T:
    token_sizes, device = broadcast_devices(token_sizes, device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    token_ptr = (token_sizes - 1)[batch_ptr] - token_ptr

    return acc_token_sizes[batch_ptr] + token_ptr


def reverse_packed_indices(batch_sizes: T, device: torch.device = None) -> T:
    batch_sizes, device = broadcast_devices(batch_sizes, device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    _, (token_ptr, batch_ptr), (token_sizes, _) = token_sizes_to_major_ptr3(batch_sizes)
    token_ptr = (token_sizes - 1)[batch_ptr] - token_ptr

    return batch_ptr + acc_batch_sizes[token_ptr]
