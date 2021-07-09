from typing import Tuple, List, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number

from torchrua.catting import cat_sequence
from torchrua.indexing import batch_sizes_to_ptr

__all__ = [
    'PaddedSequence', 'pad_sequence',
    'pad_packed_indices', 'pad_catted_sequence',
    'pad_packed_indices', 'pad_packed_sequence',
]

PaddedSequence = Union[Tensor, Tuple[Tensor, Tensor]]


def pad_sequence(sequences: List[Tensor], batch_first: bool = False,
                 padding_value: Number = 0, device: Device = None) -> Tensor:
    if device is None:
        device = sequences[0].device

    sequence, token_sizes = cat_sequence(sequences=sequences, device=device)
    return pad_catted_sequence(
        sequence=sequence, token_sizes=token_sizes,
        batch_first=batch_first, padding_value=padding_value,
    )


@torch.no_grad()
def pad_packed_indices(batch_sizes: Tensor,
                       sorted_indices: Tensor,
                       unsorted_indices: Tensor,
                       batch_first: bool,
                       device: Device = None) -> Tuple[Tuple[int, int], Tuple[Tensor, Tensor], Tensor]:
    if device is None:
        device = sorted_indices.device

    batch_sizes = batch_sizes.to(device=device)
    t = batch_sizes.size()[0]
    b = batch_sizes.max().item()

    token_ptr, batch_ptr, token_sizes = batch_sizes_to_ptr(
        batch_sizes=batch_sizes,
    )

    if sorted_indices is not None:
        batch_ptr = sorted_indices[batch_ptr]
    if unsorted_indices is not None:
        token_sizes = token_sizes[unsorted_indices]

    if batch_first:
        return (b, t), (batch_ptr, token_ptr), token_sizes
    else:
        return (t, b), (token_ptr, batch_ptr), token_sizes


def pad_packed_sequence(sequence: PackedSequence,
                        batch_first: bool = False,
                        padding_value: Number = 0,
                        device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence.data.device

    sizes, indices, token_sizes = pad_packed_indices(
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
        batch_first=batch_first,
        device=device,
    )

    data = torch.full(
        (*sizes, *sequence.data.size()[1:]), fill_value=padding_value,
        dtype=sequence.data.dtype, device=device, requires_grad=False,
    )
    data[indices] = sequence.data

    return data, token_sizes.cpu()


@torch.no_grad()
def pad_catted_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    t = token_sizes.max().item()
    b = token_sizes.size()[0]

    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(
        batch_sizes=token_sizes,
    )

    if batch_first:
        return (b, t), (batch_ptr, token_ptr)
    else:
        return (t, b), (token_ptr, batch_ptr)


def pad_catted_sequence(sequence: Tensor,
                        token_sizes: Tensor,
                        batch_first: bool = False,
                        padding_value: Number = 0,
                        device: Device = None) -> Tensor:
    if device is None:
        device = sequence.device

    sizes, indices = pad_catted_indices(
        token_sizes=token_sizes,
        batch_first=batch_first,
        device=device,
    )

    data = torch.full(
        (*sizes, *sequence.size()[1:]), fill_value=padding_value,
        dtype=sequence.dtype, device=device, requires_grad=False,
    )
    data[indices] = sequence

    return data
