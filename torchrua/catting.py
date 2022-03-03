from typing import List, NamedTuple, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import minor_sizes_to_ptr, major_sizes_to_ptr, accumulate_sizes

__all__ = [
    'CattedSequence', 'cat_sequence',
    'cat_packed_indices', 'cat_packed_sequence',
    'cat_padded_indices', 'cat_padded_sequence',
    'trunc_catted_indices', 'trunc_catted_sequence',
]


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, dtype: torch.dtype = None, device: Device = None) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(dtype=dtype, device=device),
            token_sizes=self.token_sizes.to(dtype=dtype, device=device),
        )


def cat_sequence(sequences: List[Tensor], device: Device = None) -> CattedSequence:
    if device is None:
        device = sequences[0].device

    token_sizes = torch.tensor([sequence.size()[0] for sequence in sequences], dtype=torch.long, device=device)
    return CattedSequence(
        data=torch.cat(sequences, dim=0).to(device=device),
        token_sizes=token_sizes,
    )


@torch.no_grad()
def cat_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor, device: Device = None):
    if device is None:
        if unsorted_indices is None:
            device = batch_sizes.device
        else:
            device = unsorted_indices.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr, token_sizes = minor_sizes_to_ptr(
        token_sizes=batch_sizes,
        token_ptr=unsorted_indices,
    )

    indices = acc_batch_sizes[token_ptr] + batch_ptr
    return indices, token_sizes


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
        indices = (batch_ptr, token_ptr)
    else:
        indices = (token_ptr, batch_ptr)
    return indices, token_sizes


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


@torch.no_grad()
def trunc_catted_indices(token_sizes: Tensor, trunc: Tuple[int, int], device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes = token_sizes - trunc[0] - trunc[1]
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    indices = token_ptr + trunc[0] + acc_token_sizes[batch_ptr]

    return indices, token_sizes


def trunc_catted_sequence(sequence: CattedSequence, trunc: Tuple[int, int]) -> CattedSequence:
    indices, token_sizes = trunc_catted_indices(
        token_sizes=sequence.token_sizes, trunc=trunc,
        device=sequence.data.device,
    )
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=token_sizes,
    )
