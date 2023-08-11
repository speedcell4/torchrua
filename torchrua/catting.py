from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import CattedSequence
from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import get_device
from torchrua.core import major_sizes_to_ptr
from torchrua.core import minor_sizes_to_ptr

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
    batch_ptr, token_ptr, token_sizes = minor_sizes_to_ptr(sizes=batch_sizes, minor_ptr=unsorted_indices)

    return batch_ptr + acc_batch_sizes[token_ptr], token_sizes


def cat_packed_sequence(sequence: PackedSequence, device: torch.device = None) -> CattedSequence:
    sequence, batch_sizes, sorted_indices, unsorted_indices, device = broadcast_devices(
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


def cat_padded_indices(token_sizes: Tensor, batch_first: bool, device: torch.device = None):
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    if batch_first:
        return (batch_ptr, token_ptr), token_sizes
    else:
        return (token_ptr, batch_ptr), token_sizes


def cat_padded_sequence(sequence: Tensor, token_sizes: Tensor, batch_first: bool = False, device: torch.device = None):
    sequence, token_sizes, device = broadcast_devices(sequence, token_sizes, device=device)

    indices, token_sizes = cat_padded_indices(
        token_sizes=token_sizes,
        batch_first=batch_first,
        device=device,
    )

    return CattedSequence(
        data=sequence[indices],
        token_sizes=token_sizes,
    )
