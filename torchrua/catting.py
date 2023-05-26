from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import accumulate_sizes, CattedSequence, major_sizes_to_ptr, minor_sizes_to_ptr

__all__ = [
    'cat_sequence',
    'cat_packed_indices', 'cat_packed_sequence',
    'cat_padded_indices', 'cat_padded_sequence',
]


def cat_sequence(sequences: List[Tensor], dtype: torch.dtype = None, device: Device = None) -> CattedSequence:
    if device is None:
        device = sequences[0].device

    return CattedSequence(
        data=torch.cat(sequences, dim=0).to(dtype=dtype, device=device),
        token_sizes=torch.tensor([seq.size()[0] for seq in sequences], dtype=torch.long, device=device),
    )


@torch.no_grad()
def cat_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor, device: Device = None):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        elif batch_sizes is not None:
            device = batch_sizes.device
        else:
            raise RuntimeError('batch_sizes and unsorted_indices are all None')

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr, token_sizes = minor_sizes_to_ptr(sizes=batch_sizes, minor_ptr=unsorted_indices)

    return batch_ptr + acc_batch_sizes[token_ptr], token_sizes


def cat_packed_sequence(sequence: PackedSequence, device: Device = None) -> CattedSequence:
    if device is None:
        device = sequence.data.device

    indices, token_sizes = cat_packed_indices(
        batch_sizes=sequence.batch_sizes,
        unsorted_indices=sequence.unsorted_indices,
        device=device,
    )

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=token_sizes,
    )


@torch.no_grad()
def cat_padded_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    if batch_first:
        return (batch_ptr, token_ptr), token_sizes
    else:
        return (token_ptr, batch_ptr), token_sizes


def cat_padded_sequence(sequence: Tensor, token_sizes: Tensor,
                        batch_first: bool = False, device: Device = None) -> CattedSequence:
    if device is None:
        device = sequence.device

    indices, token_sizes = cat_padded_indices(
        token_sizes=token_sizes,
        batch_first=batch_first,
        device=device,
    )

    return CattedSequence(
        data=sequence[indices],
        token_sizes=token_sizes,
    )
