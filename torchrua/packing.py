from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import cat_sequence
from torchrua.indexing import token_sizes_to_ptr
from torchrua.utils import accumulate_sizes, sizes_to_sorting_indices

__all__ = [
    'pack_sequence',
    'pack_catted_sequence',
    'pack_catted_indices', 'pack_padded_sequence',
]


def pack_sequence(sequences: List[Tensor], device: Device = None) -> PackedSequence:
    sequence, token_sizes = cat_sequence(sequences=sequences, device=device)
    return pack_catted_sequence(sequence=sequence, token_sizes=token_sizes, device=device)


def pack_padded_sequence(sequence: Tensor, token_sizes: Tensor, batch_first: bool = False) -> PackedSequence:
    with torch.no_grad():
        sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting_indices(
            sizes=token_sizes, device=sequence.device,
        )

        token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
            token_sizes=sorted_token_sizes,
            batch_ptr=sorted_indices,
        )

        if batch_first:
            index = batch_ptr, token_ptr
        else:
            index = token_ptr, batch_ptr

    return PackedSequence(
        data=sequence[index],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


@torch.no_grad()
def pack_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting_indices(
        sizes=token_sizes, device=device,
    )
    token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
        token_sizes=sorted_token_sizes,
        batch_ptr=sorted_indices,
    )
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    indices = acc_token_sizes[batch_ptr] + token_ptr

    return indices, batch_sizes, sorted_indices, unsorted_indices


def pack_catted_sequence(sequence: Tensor, token_sizes: Tensor, device: Device = None) -> PackedSequence:
    if device is None:
        device = sequence.device

    indices, batch_sizes, sorted_indices, unsorted_indices = pack_catted_indices(
        token_sizes=token_sizes, device=device,
    )

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
