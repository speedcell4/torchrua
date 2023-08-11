from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua import major_sizes_to_ptr
from torchrua.core import CattedSequence
from torchrua.core import get_device
from torchrua.core import major_sizes_to_shape

Sequence = Union[CattedSequence, PackedSequence]


def sequence_info(sequence: Sequence):
    if isinstance(sequence, CattedSequence):
        (b, t), (batch_ptr, token_ptr), token_sizes = catted_sequence_info(
            token_sizes=sequence.token_sizes,
            device=sequence.data.device,
        )
        return (b, t), (batch_ptr, token_ptr)

    if isinstance(sequence, PackedSequence):
        return packed_sequence_info(
            batch_sizes=sequence.batch_sizes,
            sorted_indices=sequence.sorted_indices,
            device=sequence.data.device,
        )

    raise TypeError(f'{type(sequence)} is not supported')


def catted_sequence_info(token_sizes: Tensor, device: torch.device = None):
    device = get_device(token_sizes, device=device)
    token_sizes = token_sizes.to(device=device)

    t, b = major_sizes_to_shape(sizes=token_sizes)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    return (b, t), (batch_ptr, token_ptr), token_sizes


def packed_sequence_info(batch_sizes: Tensor, sorted_indices: Tensor, device: torch.device = None):
    device = get_device(batch_sizes, device=device)
    batch_sizes = batch_sizes.to(device=device)

    b, t = major_sizes_to_shape(sizes=batch_sizes)
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    batch_ptr = sorted_indices[batch_ptr]

    return (b, t), (batch_ptr, token_ptr)
