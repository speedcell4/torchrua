from typing import Tuple
from typing import Union

import torch

from torchrua import broadcast_devices
from torchrua.core import accumulate_sizes
from torchrua.core import major_sizes_to_ptr
from torchrua.ty import C
from torchrua.ty import P
from torchrua.ty import T
from torchrua.ty import is_type

__all__ = [
    'trunc_sequence', 'trunc_indices',
    'trunc_catted_indices', 'trunc_packed_indices',
]


def trunc_sequence(sequence: Union[C, P], trunc: Tuple[int, int]) -> Union[C, P]:
    indices, sizes = trunc_indices(sequence, trunc=trunc)
    return type(sequence)(sequence[0][indices], sizes, *sequence[2:])


C.trunc = trunc_sequence
P.trunc = trunc_sequence


def trunc_indices(sequence: Union[C, P], trunc: Tuple[int, int], device: torch.device = None):
    if is_type(sequence, C):
        return trunc_catted_indices(
            token_sizes=sequence.token_sizes,
            trunc=trunc, device=device,
        )

    if is_type(sequence, P):
        return trunc_packed_indices(
            batch_sizes=sequence.batch_sizes,
            trunc=trunc, device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def trunc_catted_indices(token_sizes: T, trunc: Tuple[int, int], device: torch.device = None) -> Tuple[T, T]:
    token_sizes, device = broadcast_devices(token_sizes, device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes = token_sizes - trunc[0] - trunc[1]
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    return acc_token_sizes[batch_ptr] + token_ptr + trunc[0], token_sizes


def trunc_packed_indices(batch_sizes: T, trunc: Tuple[int, int], device: torch.device = None) -> Tuple[T, T]:
    batch_sizes, device = broadcast_devices(batch_sizes, device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_sizes = batch_sizes[trunc[0] + trunc[1]:]
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)

    return batch_ptr + acc_batch_sizes[token_ptr + trunc[0]], batch_sizes.cpu()
