from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import CattedSequence
from torchrua.core import broadcast_devices
from torchrua.core import major_sizes_to_ptr
from torchrua.core import major_sizes_to_shape

Sequence = Union[CattedSequence, PackedSequence]

__all__ = [
    'sequence_ptr2', 'batch_sizes_ptr2', 'token_sizes_ptr2',
]


def sequence_ptr2(sequence: Sequence):
    if isinstance(sequence, CattedSequence):
        return token_sizes_ptr2(
            token_sizes=sequence.token_sizes,
            device=sequence.data.device,
        )

    if isinstance(sequence, PackedSequence):
        return batch_sizes_ptr2(
            batch_sizes=sequence.batch_sizes,
            sorted_indices=sequence.sorted_indices,
            device=sequence.data.device,
        )

    raise TypeError(f'{type(sequence)} is not supported')


def token_sizes_ptr2(token_sizes: Tensor, device: torch.device = None):
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    t, b = major_sizes_to_shape(sizes=token_sizes)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    return (b, t), (batch_ptr, token_ptr)


def batch_sizes_ptr2(batch_sizes: Tensor, sorted_indices: Tensor, device: torch.device = None):
    sorted_indices, batch_sizes, device = broadcast_devices(sorted_indices, batch_sizes, device=device)

    b, t = major_sizes_to_shape(sizes=batch_sizes)
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    batch_ptr = sorted_indices[batch_ptr]

    return (b, t), (batch_ptr, token_ptr)
