from typing import List, Optional, NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import minor_sizes_to_ptr, major_sizes_to_ptr, accumulate_sizes


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

    token_sizes = torch.tensor([s.size()[0] for s in sequences], dtype=torch.long, device=device)
    return CattedSequence(
        data=torch.cat(sequences, dim=0).to(device=device),
        token_sizes=token_sizes,
    )


@torch.no_grad()
def cat_packed_indices(batch_sizes: Tensor, unsorted_indices: Optional[Tensor], device: Device = None):
    if device is None:
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
