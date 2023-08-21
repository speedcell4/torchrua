from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import get_device
from torchrua.info import batch_sizes_to_minor_ptr3
from torchrua.info import token_sizes_to_major_ptr3
from torchrua.ty import CattedSequence

__all__ = [
    'cat_sequence',
    'cat_packed_indices', 'cat_packed_sequence',
    'cat_padded_indices', 'cat_padded_sequence',
]


def cat_sequence(sequences: List[Tensor], dtype: torch.dtype = None, device: torch.device = None):
    device = get_device(*sequences, device=device)

    return CattedSequence(
        data=torch.cat(sequences, dim=0).to(dtype=dtype, device=device),
        token_sizes=torch.tensor([sequence.size()[0] for sequence in sequences], dtype=torch.long, device=device),
    )


def cat_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor, device: torch.device = None):
    unsorted_indices, batch_sizes, device = broadcast_devices(unsorted_indices, batch_sizes, device=device)

    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    _, (batch_ptr, token_ptr), (_, token_sizes) = batch_sizes_to_minor_ptr3(
        batch_sizes=batch_sizes, batch_ptr=unsorted_indices,
    )

    return batch_ptr + acc_batch_sizes[token_ptr], token_sizes


def cat_packed_sequence(sequence: PackedSequence, device: torch.device = None):
    sequence, batch_sizes, _, unsorted_indices, device = broadcast_devices(
        *sequence, device=device,
    )

    indices, token_sizes = cat_packed_indices(
        batch_sizes=batch_sizes,
        unsorted_indices=unsorted_indices,
        device=device,
    )

    return CattedSequence(
        data=sequence[indices],
        token_sizes=token_sizes,
    )


def cat_padded_indices(token_sizes: Tensor, device: torch.device = None):
    _, (batch_ptr, token_ptr), (_, token_sizes) = token_sizes_to_major_ptr3(token_sizes, device=device)
    return (batch_ptr, token_ptr), token_sizes


def cat_padded_sequence(sequence: Tensor, token_sizes: Tensor, device: torch.device = None):
    sequence, token_sizes, device = broadcast_devices(sequence, token_sizes, device=device)

    (batch_ptr, token_ptr), token_sizes = cat_padded_indices(
        token_sizes=token_sizes,
        device=device,
    )

    return CattedSequence(
        data=sequence[batch_ptr, token_ptr],
        token_sizes=token_sizes,
    )
