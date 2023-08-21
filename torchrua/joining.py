from typing import List
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import cat_sequence
from torchrua.core import accumulate_sizes
from torchrua.core import major_sizes_to_ptr
from torchrua.packing import pack_catted_indices
from torchrua.ty import CattedSequence


def cat_sequences(sequences) -> Union[CattedSequence, PackedSequence]:
    sequence, *_ = sequences

    if isinstance(sequence, CattedSequence):
        return cat_catted_sequences(sequences)
    if isinstance(sequence, PackedSequence):
        return cat_packed_sequences(sequences)

    raise TypeError(f'type {type(sequence)} is not supported')


@torch.no_grad()
def cat_catted_indices(token_sizes: List[Tensor], device: Device = None):
    if device is None:
        device = token_sizes[0].device

    repeats, batch_sizes = cat_sequence(token_sizes, device=device)
    batch_ptr, _ = major_sizes_to_ptr(sizes=batch_sizes)

    batch_ptr = torch.repeat_interleave(batch_ptr, repeats=repeats)
    _, indices = torch.sort(batch_ptr, stable=True, descending=False)
    _, token_sizes = torch.unique(batch_ptr, sorted=True, return_counts=True)

    return indices, token_sizes


def cat_catted_sequences(sequences: List[CattedSequence]) -> CattedSequence:
    data, token_sizes = zip(*sequences)

    indices, token_sizes = cat_catted_indices(
        token_sizes=token_sizes,
        device=data[0].device,
    )

    return CattedSequence(
        data=torch.cat(data, dim=0)[indices],
        token_sizes=token_sizes,
    )


@torch.no_grad()
def cat_packed_indices(batch_sizes: List[Tensor], sorted_indices: List[Tensor], device: Device = None):
    if device is None:
        device = sorted_indices[0].device

    batch_sizes, sizes1 = cat_sequence(batch_sizes, device=device)
    sorted_indices, sizes2 = cat_sequence(sorted_indices, device=device)

    batch_ptr0, _ = major_sizes_to_ptr(sizes=batch_sizes)

    token_ptr1 = torch.repeat_interleave(accumulate_sizes(sizes=sizes2), repeats=sizes1)
    batch_ptr1 = torch.repeat_interleave(token_ptr1, repeats=batch_sizes)
    batch_ptr0 = sorted_indices[batch_ptr0 + batch_ptr1]

    batch_ptr0, indices0 = torch.sort(batch_ptr0, stable=True)
    _, token_sizes = torch.unique(batch_ptr0, return_counts=True)
    indices1, batch_sizes, sorted_indices, unsorted_indices = pack_catted_indices(
        token_sizes=token_sizes, device=device,
    )

    return indices0[indices1], batch_sizes, sorted_indices, unsorted_indices


def cat_packed_sequences(sequences: List[PackedSequence]) -> PackedSequence:
    data, batch_sizes, sorted_indices, _ = zip(*sequences)

    indices, batch_sizes, sorted_indices, unsorted_indices = cat_packed_indices(
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        device=data[0].device,
    )

    return PackedSequence(
        data=torch.cat(data, dim=0)[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def stack_catted_sequences(sequences: List[CattedSequence]) -> CattedSequence:
    return CattedSequence(
        data=torch.cat([sequence.data for sequence in sequences], dim=0),
        token_sizes=torch.cat([sequence.token_sizes for sequence in sequences], dim=0),
    )


def stack_packed_sequences(sequences: List[PackedSequence]) -> PackedSequence:
    raise NotImplementedError
